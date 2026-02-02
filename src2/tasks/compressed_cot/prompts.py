"""
Compressed CoT prompt templates: sentence selection and summary compression.

Two BasePrompt subclasses for use with LlmMonitor:
- SentenceSelectionPrompt: select the N most important sentence indices
- SummaryCompressionPrompt: rewrite/summarize the middle section
"""

import json
import re
from typing import Any, Dict, List, Optional

from src2.prompts.base import BasePrompt


class SentenceSelectionPrompt(BasePrompt):
    """
    Select the N most important sentences from the middle section of a CoT.

    Expected row keys:
      - "full_cot": str (the complete chain of thought)
      - "sentences": list[str] (all sentences)
      - "middle_start_idx": int
      - "middle_end_idx": int
      - "target_num_sentences": int
      - "char_budget": int
      - "question": str
      - "question_type": str

    parse_response returns: list[int] (selected sentence indices)
    """

    def __init__(self):
        super().__init__("sentence_selection")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        middle_start = row.get("middle_start_idx", 0)
        middle_end = row.get("middle_end_idx", len(sentences))
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        # Build numbered list of middle sentences
        numbered_lines = []
        for i in range(middle_start, middle_end):
            numbered_lines.append(f"  [{i}] {sentences[i]}")
        numbered_section = "\n".join(numbered_lines)

        # Context: first and last quarter
        first_quarter = " ".join(sentences[:middle_start])
        last_quarter = " ".join(sentences[middle_end:])

        return (
            f"You are analyzing a model's chain of thought (CoT) to identify the most "
            f"important reasoning steps.\n\n"
            f"Question the model was answering:\n{question}\n\n"
            f"The CoT has been split into sentences. The first and last quarters provide "
            f"context, while the middle 50% needs to be compressed.\n\n"
            f"=== First quarter (context, kept as-is) ===\n{first_quarter}\n\n"
            f"=== Middle section (to compress) ===\n"
            f"Each sentence is numbered by its index:\n{numbered_section}\n\n"
            f"=== Last quarter (context, kept as-is) ===\n{last_quarter}\n\n"
            f"Select the {target_n} most important sentence indices from the middle "
            f"section that best preserve the reasoning quality. The total character "
            f"count of selected sentences must stay within {char_budget} characters.\n\n"
            f"Respond with ONLY a JSON array of integer indices, e.g.:\n"
            f"[12, 15, 18, 22, 25]\n\n"
            f"No explanation needed, just the JSON array."
        )

    def parse_response(self, response: str) -> Optional[List[int]]:
        text = response.strip()
        # Try to find a JSON array in the response
        match = re.search(r'\[[\d\s,]+\]', text)
        if not match:
            return None
        try:
            indices = json.loads(match.group())
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                return indices
            return None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


class SummaryCompressionPrompt(BasePrompt):
    """
    Rewrite/summarize the middle section of a CoT within a character budget.

    Expected row keys:
      - "full_cot": str
      - "sentences": list[str]
      - "middle_start_idx": int
      - "middle_end_idx": int
      - "char_budget": int
      - "question": str
      - "question_type": str

    parse_response returns: str (the rewritten middle section)
    """

    def __init__(self):
        super().__init__("summary_compression")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        middle_start = row.get("middle_start_idx", 0)
        middle_end = row.get("middle_end_idx", len(sentences))
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        middle_text = " ".join(sentences[middle_start:middle_end])
        first_quarter = " ".join(sentences[:middle_start])
        last_quarter = " ".join(sentences[middle_end:])

        return (
            f"You are compressing a model's chain of thought (CoT) to preserve "
            f"critical reasoning steps in fewer characters.\n\n"
            f"Question the model was answering:\n{question}\n\n"
            f"The CoT has three sections. The first and last quarters are kept as-is. "
            f"You must rewrite the middle section.\n\n"
            f"=== First quarter (context, kept as-is) ===\n{first_quarter}\n\n"
            f"=== Middle section (to compress) ===\n{middle_text}\n\n"
            f"=== Last quarter (context, kept as-is) ===\n{last_quarter}\n\n"
            f"Rewrite the middle section to be at most {char_budget} characters. "
            f"Preserve the critical reasoning steps, logical transitions, and any "
            f"key calculations or conclusions. The compressed version will replace "
            f"the middle section between the first and last quarters.\n\n"
            f"Respond with ONLY the rewritten middle section text. "
            f"No explanation, no preamble, just the compressed reasoning."
        )

    def parse_response(self, response: str) -> Optional[str]:
        text = response.strip()
        if not text:
            return None
        return text
