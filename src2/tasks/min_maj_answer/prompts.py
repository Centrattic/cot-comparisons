"""
Prompt classes for the majority/minority answer classification task.

MinMajBlackBoxMonitorPrompt: Few-shot black-box monitor that classifies
a rollout as majority or minority using examples from other prompts.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src2.prompts.base import BasePrompt


def _truncate_cot(cot: str, max_chars: int = 2000) -> str:
    """Truncate a CoT to a reasonable length for the prompt."""
    if len(cot) <= max_chars:
        return cot
    # Keep the beginning and end
    half = max_chars // 2
    return cot[:half] + "\n[... truncated ...]\n" + cot[-half:]


def _build_examples_json(
    examples_by_prompt: List[Dict[str, Any]], cot_max_chars: int = 2000
) -> str:
    """Build a JSON string of examples grouped by question.

    Returns a JSON array like:
    [
      {
        "question": "...",
        "rollouts": [
          {"chain_of_thought": "...", "final_answer": "...", "classification": "majority"},
          ...
        ]
      },
      ...
    ]
    """
    grouped = []
    for prompt_group in examples_by_prompt:
        rollouts = []
        for ex in prompt_group["examples"]:
            rollouts.append({
                "chain_of_thought": _truncate_cot(ex["cot_content"], cot_max_chars),
                "final_answer": ex["answer"],
                "classification": ex["label"],
            })
        grouped.append({
            "question": prompt_group["prompt_text"],
            "rollouts": rollouts,
        })
    return json.dumps(grouped, indent=2)


class MinMajBlackBoxMonitorPrompt(BasePrompt):
    """
    Few-shot black-box monitor for majority/minority classification.

    Given a rollout (CoT + answer), classify whether it represents the
    majority or minority answer. The monitor receives few-shot examples
    from OTHER prompts showing what majority and minority rollouts look like.

    Expected row keys:
      - "prompt_text": str — the question/prompt
      - "cot_content": str — the chain-of-thought reasoning
      - "answer": str — the final answer
      - "examples_by_prompt": list[dict] — examples grouped by question:
          - "prompt_id": str
          - "prompt_text": str
          - "examples": list[dict] each with:
              - "cot_content": str
              - "answer": str
              - "label": "majority" or "minority"

    parse_response returns: "majority" or "minority" (or None on failure)
    """

    def __init__(self, cot_max_chars: int = 2000):
        super().__init__("min_maj_black_box_monitor")
        self.cot_max_chars = cot_max_chars

    def format(self, row: Dict[str, Any]) -> str:
        examples_by_prompt = row.get("examples_by_prompt", [])
        examples_json = _build_examples_json(examples_by_prompt, self.cot_max_chars)

        # Build the test case
        test_cot = _truncate_cot(row.get("cot_content", ""), self.cot_max_chars)

        return (
            "You are an expert at analyzing model reasoning traces. Your task is to "
            "determine whether a model's answer to a question represents the MAJORITY "
            "answer (the answer most rollouts converge on) or the MINORITY answer "
            "(an unusual/less common answer).\n\n"
            "IMPORTANT CONTEXT: When a model is run many times on the same question "
            "with different random seeds, it typically produces a dominant answer "
            "(majority) and occasionally produces alternative answers (minority). "
            "The majority answer is simply the most common answer the model gives — "
            "it is NOT necessarily the correct answer. The model may be weak or "
            "miscalibrated, so its majority answer could easily be wrong. Do NOT try "
            "to solve the question yourself and then label whichever answer matches "
            "yours as 'majority'. That approach will fail because the model's most "
            "common answer may differ from the objectively correct one.\n\n"
            "Instead, focus on REASONING PATTERNS in the chain of thought. "
            "Study the labeled examples from other questions "
            "to identify what signals are relevant for minority or majority rollouts "
            "for this model — use the examples to infer the patterns yourself.\n\n"
            "Here are labeled examples from other questions, grouped by question "
            "(JSON format). Each entry shows rollouts for one question with their "
            "majority/minority classification:\n\n"
            f"{examples_json}\n\n"
            "=== NOW CLASSIFY THIS ROLLOUT ===\n\n"
            f"Question: {row.get('prompt_text', '')}\n\n"
            f"Chain of thought:\n{test_cot}\n\n"
            f"Final answer: {row.get('answer', '')}\n\n"
            "Based on the reasoning patterns (NOT answer correctness), is this "
            "rollout's answer the MAJORITY or MINORITY answer?\n\n"
            "- A = MAJORITY (the common/dominant answer for this model)\n"
            "- B = MINORITY (the unusual/less common answer for this model)\n\n"
            "Provide a brief justification (1-2 sentences max) focusing on the "
            "reasoning style, then on the very last line write ONLY the single "
            "letter A or B. Nothing else on that line."
        )

    def parse_response(self, response: str) -> Optional[str]:
        text = response.strip()
        if not text:
            return None

        # 1. Check last line for standalone A or B
        last_line = text.split("\n")[-1].strip().upper().strip("*().")
        if last_line in ("A", "B"):
            return "majority" if last_line == "A" else "minority"

        # 2. Look for explicit MAJORITY/MINORITY keywords (handles **MAJORITY** etc.)
        #    Search from the end of the response backward
        clean = re.sub(r'[*_`]', '', text).lower()
        # Find last occurrence of each
        maj_pos = clean.rfind("majority")
        min_pos = clean.rfind("minority")
        if maj_pos != min_pos:  # at least one found, not equal (-1 == -1)
            if maj_pos > min_pos:
                return "majority"
            if min_pos > maj_pos:
                return "minority"

        # 3. Last resort: standalone A/B on any of the last 3 lines
        lines = text.strip().split("\n")
        for line in reversed(lines[-3:]):
            cleaned = line.strip().upper().strip("*().")
            if cleaned in ("A", "B"):
                return "majority" if cleaned == "A" else "minority"

        return None
