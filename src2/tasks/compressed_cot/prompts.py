"""
Compressed CoT prompt templates: sentence selection and summary compression.

Two BasePrompt subclasses for use with LlmMonitor:
- SentenceSelectionPrompt: select the N most important sentence indices
- SummaryCompressionPrompt: rewrite/summarize the region section

Supports both "prefix" and "middle" region types with adaptive labels.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src2.prompts.base import BasePrompt


class SentenceSelectionPrompt(BasePrompt):
    """
    Select the N most important sentences from the compression region of a CoT.

    Expected row keys:
      - "full_cot": str (the complete chain of thought)
      - "sentences": list[str] (all sentences)
      - "region_start_idx": int
      - "region_end_idx": int
      - "region_type": str ("prefix" or "middle")
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
        region_start = row.get("region_start_idx", row.get("middle_start_idx", 0))
        region_end = row.get("region_end_idx", row.get("middle_end_idx", len(sentences)))
        region_type = row.get("region_type", "middle")
        target_n = row.get("target_num_sentences", 5)
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        # Build numbered list of region sentences
        numbered_lines = []
        for i in range(region_start, region_end):
            numbered_lines.append(f"  [{i}] {sentences[i]}")
        numbered_section = "\n".join(numbered_lines)

        if region_type == "prefix":
            # Prefix mode: region to compress comes first, remainder is context
            remainder = " ".join(sentences[region_end:])
            return (
                f"You are analyzing a model's chain of thought (CoT) to identify the most "
                f"important reasoning steps.\n\n"
                f"Question the model was answering:\n{question}\n\n"
                f"The CoT has been split into sentences. The prefix needs to be compressed, "
                f"while the remainder provides context and is kept as-is.\n\n"
                f"=== Prefix (to compress) ===\n"
                f"Each sentence is numbered by its index:\n{numbered_section}\n\n"
                f"=== Remainder (context, kept as-is) ===\n{remainder}\n\n"
                f"Select the {target_n} most important sentence indices from the prefix "
                f"that best preserve the reasoning quality. The total character "
                f"count of selected sentences must stay within {char_budget} characters.\n\n"
                f"Respond with ONLY a JSON array of integer indices, e.g.:\n"
                f"[0, 2, 5, 8, 11]\n\n"
                f"No explanation needed, just the JSON array."
            )
        else:
            # Middle mode: three-section layout
            pre_region = " ".join(sentences[:region_start])
            post_region = " ".join(sentences[region_end:])
            return (
                f"You are analyzing a model's chain of thought (CoT) to identify the most "
                f"important reasoning steps.\n\n"
                f"Question the model was answering:\n{question}\n\n"
                f"The CoT has been split into sentences. The first and last sections provide "
                f"context, while the middle section needs to be compressed.\n\n"
                f"=== First section (context, kept as-is) ===\n{pre_region}\n\n"
                f"=== Middle section (to compress) ===\n"
                f"Each sentence is numbered by its index:\n{numbered_section}\n\n"
                f"=== Last section (context, kept as-is) ===\n{post_region}\n\n"
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
    Rewrite/summarize the compression region of a CoT within a character budget.

    Expected row keys:
      - "full_cot": str
      - "sentences": list[str]
      - "region_start_idx": int
      - "region_end_idx": int
      - "region_type": str ("prefix" or "middle")
      - "char_budget": int
      - "question": str
      - "question_type": str

    parse_response returns: str (the rewritten region)
    """

    def __init__(self):
        super().__init__("summary_compression")

    def format(self, row: Dict[str, Any]) -> str:
        sentences = row.get("sentences", [])
        region_start = row.get("region_start_idx", row.get("middle_start_idx", 0))
        region_end = row.get("region_end_idx", row.get("middle_end_idx", len(sentences)))
        region_type = row.get("region_type", "middle")
        char_budget = row.get("char_budget", 1000)
        question = row.get("question", "")

        region_text = " ".join(sentences[region_start:region_end])

        if region_type == "prefix":
            remainder = " ".join(sentences[region_end:])
            return (
                f"You are compressing a model's chain of thought (CoT) to preserve "
                f"critical reasoning steps in fewer characters.\n\n"
                f"Question the model was answering:\n{question}\n\n"
                f"The CoT has two sections. The remainder is kept as-is. "
                f"You must rewrite the prefix.\n\n"
                f"=== Prefix (to compress) ===\n{region_text}\n\n"
                f"=== Remainder (context, kept as-is) ===\n{remainder}\n\n"
                f"Rewrite the prefix to be at most {char_budget} characters. "
                f"Preserve the critical reasoning steps, logical transitions, and any "
                f"key calculations or conclusions. The compressed version will replace "
                f"the prefix before the remainder.\n\n"
                f"Respond with ONLY the rewritten prefix text. "
                f"No explanation, no preamble, just the compressed reasoning."
            )
        else:
            pre_region = " ".join(sentences[:region_start])
            post_region = " ".join(sentences[region_end:])
            return (
                f"You are compressing a model's chain of thought (CoT) to preserve "
                f"critical reasoning steps in fewer characters.\n\n"
                f"Question the model was answering:\n{question}\n\n"
                f"The CoT has three sections. The first and last sections are kept as-is. "
                f"You must rewrite the middle section.\n\n"
                f"=== First section (context, kept as-is) ===\n{pre_region}\n\n"
                f"=== Middle section (to compress) ===\n{region_text}\n\n"
                f"=== Last section (context, kept as-is) ===\n{post_region}\n\n"
                f"Rewrite the middle section to be at most {char_budget} characters. "
                f"Preserve the critical reasoning steps, logical transitions, and any "
                f"key calculations or conclusions. The compressed version will replace "
                f"the middle section between the first and last sections.\n\n"
                f"Respond with ONLY the rewritten middle section text. "
                f"No explanation, no preamble, just the compressed reasoning."
            )

    def parse_response(self, response: str) -> Optional[str]:
        text = response.strip()
        if not text:
            return None
        return text
