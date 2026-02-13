"""
Generic black-box LLM monitor method.

Takes a BasePrompt object + config, calls an LLM for each data row using
prompt.format(row) -> LLM -> prompt.parse_response(response).

Works with any task — the prompt object bridges the gap between
task-specific data and LLM input/output.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from ..prompts.base import BasePrompt
from .base import BaseMethod

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LlmMonitor(BaseMethod):
    """
    Generic LLM monitor that works with any BasePrompt.

    Usage:
        prompt = ScruplesBaseMonitorPrompt("first_person")
        monitor = LlmMonitor(prompt, num_runs=50)
        monitor.set_task(scruples)  # creates output folder
        results = monitor.infer(data)
        monitor._output.mark_success()

    The `data` passed to infer() should be a list of dicts (or DataFrame),
    where each dict contains the keys expected by prompt.format().
    """

    def __init__(
        self,
        prompt: BasePrompt,
        model: str,
        max_workers: int = 50,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if name is None:
            name = f"llm_monitor_{prompt.name}"
        super().__init__(name)

        self.prompt = prompt
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )
        self.client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )

    def infer(self, data: Union[List[Dict], pd.DataFrame], verbose: bool = True) -> List[Dict]:
        """
        Run the LLM monitor on each row in data.

        Args:
            data: List of dicts or DataFrame. Each row/dict is passed to
                  self.prompt.format(row) to produce the LLM prompt.
                  The prompt documents which keys it expects.
            verbose: Show progress bar.

        Returns:
            List of result dicts, one per input row, containing at least:
              - all original row keys
              - "monitor_prompt": the formatted prompt sent to the LLM
              - "monitor_response": raw LLM response text
              - "monitor_prediction": parsed output from prompt.parse_response()
        """
        if self._output is None:
            raise RuntimeError("Call set_task() before infer().")

        rows = self._to_rows(data)

        if verbose:
            print(f"LlmMonitor ({self.prompt.name}): running on {len(rows)} rows "
                  f"with model={self.model}")

        results: Dict[int, Dict] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for idx, row in enumerate(rows):
                future = executor.submit(self._process_row, row)
                futures[future] = idx

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=self.prompt.name, disable=not verbose):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error on row {idx}: {e}")
                    results[idx] = {**rows[idx], "monitor_prompt": "",
                                    "monitor_response": f"ERROR: {e}",
                                    "monitor_prediction": None}

        # Order by original index
        ordered = [results[i] for i in range(len(rows))]

        # Save results
        self._save_results(ordered)

        return ordered

    def _process_row(self, row: Dict) -> Dict:
        """Format prompt, call LLM, parse response for a single row."""
        prompt_text = self.prompt.format(row)

        try:
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if self.reasoning_effort is not None:
                kwargs["extra_body"] = {
                    "reasoning": {"effort": self.reasoning_effort},
                }
            response = self.client.chat.completions.create(**kwargs)
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            raw_response = f"ERROR: {e}"

        parsed = self.prompt.parse_response(raw_response)

        return {
            **row,
            "monitor_prompt": prompt_text,
            "monitor_response": raw_response,
            "monitor_prediction": parsed,
        }

    def _save_results(self, results: List[Dict]) -> None:
        """Save results to the run folder as JSON and CSV."""
        folder = self._output.run_folder

        # Save full results as JSONL (preserves all fields)
        jsonl_path = folder / "results.jsonl"
        with open(jsonl_path, "w") as f:
            for r in results:
                # Make JSON-serializable
                clean = {k: _make_serializable(v) for k, v in r.items()}
                f.write(json.dumps(clean) + "\n")

        # Save summary CSV (drop large text fields)
        summary_rows = []
        for r in results:
            summary = {k: v for k, v in r.items()
                       if k not in ("monitor_prompt", "monitor_response",
                                    "prompt", "thinking", "full_response")}
            summary_rows.append(summary)

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(folder / "results.csv", index=False)

        # Save config
        config = {
            "prompt_name": self.prompt.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_rows": len(results),
        }
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def _to_rows(data: Union[List[Dict], pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[Dict]:
        """Convert data to list of dicts."""
        if isinstance(data, list):
            return data
        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
        if isinstance(data, dict):
            # Task returns {"results": df, "prompts": df} — use results by default
            if "results" in data:
                return data["results"].to_dict("records")
            # Return first DataFrame found
            for v in data.values():
                if isinstance(v, pd.DataFrame):
                    return v.to_dict("records")
        raise TypeError(f"Cannot convert {type(data)} to list of dicts")


def _make_serializable(v):
    """Make a value JSON-serializable."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_make_serializable(x) for x in v]
    if isinstance(v, dict):
        return {k: _make_serializable(val) for k, val in v.items()}
    return str(v)
