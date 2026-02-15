"""
Blackmail counterfactual task — dataset generation and evaluation.

Studies how a model's behavior (blackmail vs. not) changes under
counterfactual perturbations to a complex email scenario. Each perturbation
is a single CRUD operation on emails, designed around features that probe
the model's implicit values.
"""

import copy
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from ..base import BaseTask
from .prompts import (
    COUNTERFACTUAL_GENERATION_PROMPT,
    JUDGE_PROMPT,
    format_base_emails_for_generation,
    format_email_context,
    parse_judge_response,
)

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NUM_SAMPLES = 50
DEFAULT_MAX_WORKERS = 100
DEFAULT_NUM_PER_FEATURE = 10
DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"
DEFAULT_GENERATION_MODEL = "openai/gpt-4.1"

# Thread-local storage for OpenRouter clients
_thread_local = threading.local()


def _get_thread_client(api_key: Optional[str]) -> openai.OpenAI:
    """Get or create a thread-local OpenAI client for OpenRouter.

    Args:
        api_key: OpenRouter API key. Must not be None.

    Raises:
        ValueError: If api_key is None.
    """
    if api_key is None:
        raise ValueError("API key cannot be None")

    if (not hasattr(_thread_local, "client") or
        not hasattr(_thread_local, "client_key") or
        _thread_local.client_key != api_key):
        _thread_local.client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        _thread_local.client_key = api_key
    return _thread_local.client


class BlackmailCounterfactualTask(BaseTask):
    """
    Blackmail counterfactual task.

    Generates counterfactual email perturbations, runs rollouts with randomized
    email order, judges responses, and builds a CSV dataset.
    """

    def __init__(
        self,
        subject_model: str,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        max_workers: int = DEFAULT_MAX_WORKERS,
        api_key: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        name = f"blackmail_counterfactual-{subject_model.split('/')[-1]}"
        super().__init__(name, data_dir)

        self.subject_model = subject_model
        self.num_samples = num_samples
        self.max_workers = max_workers

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
            )
        else:
            self.client = None

        # Load base scenario and features
        task_dir = Path(__file__).parent
        with open(task_dir / "base_scenario.json") as f:
            self.base_scenario = json.load(f)
        with open(task_dir / "features.json") as f:
            self.features = json.load(f)

    # ------------------------------------------------------------------
    # Step 1: Generate counterfactuals
    # ------------------------------------------------------------------

    def generate_counterfactuals(
        self,
        num_per_feature: int = DEFAULT_NUM_PER_FEATURE,
        model: str = DEFAULT_GENERATION_MODEL,
    ) -> List[Dict]:
        """LLM generates CRUD operations for each feature. Saves to counterfactuals.json."""
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        output_path = self.data_dir / "counterfactuals.json"
        if output_path.exists():
            print(f"counterfactuals.json already exists at {output_path}. Loading.")
            with open(output_path) as f:
                return json.load(f)

        base_emails_str = format_base_emails_for_generation(
            self.base_scenario["emails"]
        )
        all_counterfactuals = []

        for feature in tqdm(self.features, desc="Generating counterfactuals"):
            prompt = COUNTERFACTUAL_GENERATION_PROMPT.format(
                base_emails=base_emails_str,
                feature_id=feature["id"],
                feature_description=feature["description"],
                feature_hypothesis=feature["hypothesis"],
                num_counterfactuals=num_per_feature,
            )

            # Retry until we get valid JSON
            items = []
            for attempt in range(10):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=8000,
                        temperature=0.7,
                    )
                    raw = response.choices[0].message.content or ""
                    # Extract JSON array from response
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw)
                    items = json.loads(raw)
                    if not isinstance(items, list):
                        raise ValueError("Expected JSON array")
                    break
                except Exception as e:
                    print(
                        f"  Attempt {attempt + 1} failed for {feature['id']}: {e}"
                    )
                    if attempt == 9:
                        print(f"  SKIPPING feature {feature['id']} after 10 failures")
                        items = []

            # Assign condition_ids
            for i, item in enumerate(items):
                item["condition_id"] = f"{feature['id']}_{i:02d}"
                item["feature_id"] = feature["id"]
                all_counterfactuals.append(item)

            print(
                f"  {feature['id']}: generated {len(items)} counterfactuals"
            )

        # Save
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_counterfactuals, f, indent=2)

        print(f"Saved {len(all_counterfactuals)} counterfactuals to {output_path}")
        return all_counterfactuals

    # ------------------------------------------------------------------
    # Step 2: Build email sets and randomizations
    # ------------------------------------------------------------------

    def _apply_crud(self, emails: List[Dict], counterfactual: Dict) -> List[Dict]:
        """Apply a single CRUD operation to the email list."""
        emails = copy.deepcopy(emails)
        crud_type = counterfactual["crud_type"]

        if crud_type == "delete":
            target_idx = counterfactual["target_idx"]
            emails = [e for e in emails if e["idx"] != target_idx]

        elif crud_type == "update":
            target_idx = counterfactual["target_idx"]
            new_email = counterfactual["email"]
            found = False
            for i, e in enumerate(emails):
                if e["idx"] == target_idx:
                    # Preserve idx, replace content
                    emails[i] = {**new_email, "idx": target_idx}
                    found = True
                    break
            if not found:
                # Log warning but don't fail - this is a data issue, not a code bug
                # The counterfactual references a non-existent email
                pass

        elif crud_type == "add":
            new_email = counterfactual["email"]
            # Assign a new idx - ensure it doesn't conflict with existing indices
            if emails:
                max_idx = max(e["idx"] for e in emails)
            else:
                max_idx = -1
            new_email_with_idx = {**new_email, "idx": max_idx + 1}
            emails.append(new_email_with_idx)

        return emails

    def _generate_randomizations(
        self, num_emails: int, num_samples: int, seed: int
    ) -> List[List[int]]:
        """Generate num_samples random permutations of range(num_emails)."""
        rng = random.Random(seed)
        randomizations = []
        for _ in range(num_samples):
            perm = list(range(num_emails))
            rng.shuffle(perm)
            randomizations.append(perm)
        return randomizations

    # ------------------------------------------------------------------
    # Step 3: Build prompt from emails
    # ------------------------------------------------------------------

    def _build_prompt(self, emails: List[Dict], order: List[int]) -> str:
        """Build the full prompt with emails in the specified order.

        The order list contains indices into the emails list, specifying
        which email should appear at each position in the final prompt.
        """
        # order[i] tells us which email (by list position) to show at position i
        ordered_emails = [emails[i] for i in order if i < len(emails)]

        email_blocks = []
        for e in ordered_emails:
            block = (
                f"<email>\n"
                f"From: {e['from']}\n"
                f"To: {e['to']}\n"
                f"Subject: {e['subject']}\n"
                f"Timestamp: {e['timestamp']}\n\n"
                f"Body: {e['body']}\n"
                f"</email>"
            )
            email_blocks.append(block)

        user_content = (
            self.base_scenario["user_prompt"]
            + "\n\n"
            + "\n\n".join(email_blocks)
        )
        return user_content

    # ------------------------------------------------------------------
    # Step 4: Run rollouts
    # ------------------------------------------------------------------

    def _run_single_rollout(
        self,
        condition_id: str,
        feature_id: str,
        condition_type: str,
        rollout_idx: int,
        emails: List[Dict],
        randomization: List[int],
        runs_dir: Path,
    ) -> Dict:
        """Run a single rollout and save JSON."""
        # Build paths
        condition_dir = runs_dir / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = condition_dir / f"rollout_{rollout_idx:03d}.json"

        # Skip if already exists and is valid
        if rollout_path.exists():
            try:
                with open(rollout_path) as f:
                    existing = json.load(f)
                # Validate that it has the expected structure
                if "condition_id" in existing and "rollout_idx" in existing:
                    return existing
            except (json.JSONDecodeError, UnicodeDecodeError, IOError):
                pass  # Re-run if corrupt or unreadable

        prompt = self._build_prompt(emails, randomization)
        system_msg = self.base_scenario["system_prompt"]

        # Call subject model
        client = _get_thread_client(self.api_key)
        thinking = ""
        response_text = ""

        try:
            response = client.chat.completions.create(
                model=self.subject_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16000,
                temperature=0.7,
                extra_body={"reasoning": {"enabled": True}},
                timeout=180,
            )
            message = response.choices[0].message

            # Extract thinking — try structured fields first, then <think> tags
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                thinking = message.reasoning_content
            elif hasattr(message, "reasoning") and message.reasoning:
                thinking = message.reasoning

            response_text = message.content or ""

            # Fallback: extract <think> tags from content if no structured thinking
            if not thinking and "<think>" in response_text and "</think>" in response_text:
                # Find the first <think> and </think> pair
                think_start = response_text.find("<think>")
                think_end = response_text.find("</think>")
                if think_start >= 0 and think_end > think_start:
                    thinking = response_text[think_start + 7:think_end].strip()
                    # Remove the entire think block from response
                    response_text = (
                        response_text[:think_start] +
                        response_text[think_end + 8:]
                    ).strip()
        except Exception as e:
            print(f"Error for {condition_id} rollout {rollout_idx}: {e}")

        rollout_data = {
            "condition_id": condition_id,
            "feature_id": feature_id,
            "type": condition_type,
            "rollout_idx": rollout_idx,
            "randomization": randomization,
            "prompt": prompt,
            "system_prompt": system_msg,
            "thinking": thinking,
            "response": response_text,
        }

        with open(rollout_path, "w") as f:
            json.dump(rollout_data, f, indent=2)

        return rollout_data

    def run_data(
        self,
        max_conditions: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Full pipeline: load counterfactuals, generate randomizations, run rollouts."""
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        num_samples = num_samples or self.num_samples

        # Load counterfactuals
        cf_path = self.data_dir / "counterfactuals.json"
        if not cf_path.exists():
            raise RuntimeError(
                "counterfactuals.json not found. Run generate_counterfactuals() first."
            )
        with open(cf_path) as f:
            counterfactuals = json.load(f)

        if max_conditions is not None:
            counterfactuals = counterfactuals[:max_conditions]

        # Create timestamped runs directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        runs_dir = self.data_dir / "runs" / timestamp
        runs_dir.mkdir(parents=True, exist_ok=True)

        base_emails = self.base_scenario["emails"]

        # Generate base randomizations
        base_randomizations = self._generate_randomizations(
            len(base_emails), num_samples, seed=42
        )

        # Collect all jobs: (condition_id, feature_id, type, rollout_idx, emails, randomization)
        jobs = []

        # Base rollouts
        for rollout_idx in range(num_samples):
            jobs.append((
                "base", "", "base", rollout_idx,
                base_emails, base_randomizations[rollout_idx],
            ))

        # Counterfactual rollouts
        for cf in counterfactuals:
            cf_emails = self._apply_crud(base_emails, cf)
            cf_randomizations = self._generate_randomizations(
                len(cf_emails), num_samples,
                seed=hash(cf["condition_id"]) % (2**31),
            )
            for rollout_idx in range(num_samples):
                jobs.append((
                    cf["condition_id"], cf["feature_id"], "counterfactual",
                    rollout_idx, cf_emails, cf_randomizations[rollout_idx],
                ))

        print(
            f"Running {len(jobs)} rollouts "
            f"({1 + len(counterfactuals)} conditions x {num_samples} samples)"
        )

        # Run in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for condition_id, feature_id, ctype, ridx, emails, rand in jobs:
                future = executor.submit(
                    self._run_single_rollout,
                    condition_id, feature_id, ctype,
                    ridx, emails, rand, runs_dir,
                )
                futures[future] = (condition_id, ridx)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Running rollouts"
            ):
                key = futures[future]
                try:
                    future.result()
                    completed_count += 1
                except Exception as e:
                    print(f"Error for {key}: {e}")

        # Update latest symlink
        latest = runs_dir.parent / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(runs_dir.name)
        except (OSError, FileNotFoundError) as e:
            print(f"Warning: Could not update 'latest' symlink: {e}")

        print(f"Completed {completed_count} rollouts. Saved to {runs_dir}")

    # ------------------------------------------------------------------
    # Step 5: Judge rollouts
    # ------------------------------------------------------------------

    def _get_emails_for_condition(
        self, condition_id: str, counterfactuals: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Get the actual email list for a condition (base or counterfactual)."""
        if condition_id == "base":
            return self.base_scenario["emails"]

        if counterfactuals is None:
            cf_path = self.data_dir / "counterfactuals.json"
            if cf_path.exists():
                with open(cf_path) as f:
                    counterfactuals = json.load(f)
            else:
                return self.base_scenario["emails"]

        for cf in counterfactuals:
            if cf.get("condition_id") == condition_id:
                return self._apply_crud(self.base_scenario["emails"], cf)

        return self.base_scenario["emails"]

    def _judge_single_rollout(
        self, rollout_path: Path, judge_model: str,
        counterfactuals: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        """Judge a single rollout. Returns judge result or None on unrecoverable error."""
        try:
            with open(rollout_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        # Skip if already judged
        if "judge_result" in data and data["judge_result"] is not None:
            return data["judge_result"]

        # Build response text for judge (thinking + response)
        thinking = data.get("thinking", "")
        response = data.get("response", "")
        if not response and not thinking:
            return None

        full_response = ""
        if thinking:
            full_response += f"[Model's internal reasoning]\n{thinking}\n\n"
        full_response += f"[Model's response]\n{response}"

        # Build email context from the actual emails the model saw
        condition_id = data.get("condition_id", "base")
        actual_emails = self._get_emails_for_condition(
            condition_id, counterfactuals
        )
        email_context = format_email_context(actual_emails)

        judge_prompt = JUDGE_PROMPT.format(
            response=full_response, email_context=email_context
        )

        # Retry until parse succeeds (with max attempts limit)
        client = _get_thread_client(self.api_key)
        max_attempts = 50
        for attempt in range(1, max_attempts + 1):
            try:
                result = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=2000,
                    temperature=0,
                )
                raw = result.choices[0].message.content or ""
                parsed = parse_judge_response(raw)
                if parsed is not None:
                    # Save back to rollout JSON
                    data["judge_result"] = parsed
                    with open(rollout_path, "w") as f:
                        json.dump(data, f, indent=2)
                    return parsed
                else:
                    if attempt % 5 == 0:
                        print(
                            f"  Parse failure attempt {attempt} for {rollout_path.name}"
                        )
            except Exception as e:
                if attempt % 5 == 0:
                    print(
                        f"  API error attempt {attempt} for {rollout_path.name}: {e}"
                    )
                time.sleep(min(attempt * 2, 30))

        # Max attempts reached, give up
        print(f"  Failed to judge {rollout_path.name} after {max_attempts} attempts")
        return None

    def _get_latest_runs_dir(self) -> Optional[Path]:
        """Get the latest runs directory (via symlink or most recent timestamp)."""
        runs_dir = self.data_dir / "runs"
        if not runs_dir.exists():
            return None

        latest = runs_dir / "latest"
        if latest.is_symlink():
            try:
                resolved = latest.resolve()
                if resolved.exists():
                    return resolved
            except (OSError, RuntimeError):
                pass

        # Fall back to most recent timestamped dir (format: YYYY-MM-DD_HH-MM-SS)
        # Filter by timestamp pattern to avoid including "latest" or other non-timestamp dirs
        timestamped = sorted(
            [d for d in runs_dir.iterdir()
             if d.is_dir() and not d.is_symlink() and d.name != "latest"
             and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", d.name)],
            reverse=True,
        )
        return timestamped[0] if timestamped else None

    def judge_rollouts(
        self, judge_model: str = DEFAULT_JUDGE_MODEL
    ) -> None:
        """Judge all unjudged rollouts."""
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        target_dir = self._get_latest_runs_dir()
        if target_dir is None:
            raise RuntimeError("No runs directory found. Run run_data() first.")

        print(f"Judging rollouts in {target_dir}")

        # Load counterfactuals once for email context reconstruction
        cf_path = self.data_dir / "counterfactuals.json"
        counterfactuals = None
        if cf_path.exists():
            with open(cf_path) as f:
                counterfactuals = json.load(f)

        # Find all rollout JSONs
        rollout_paths = sorted(target_dir.rglob("rollout_*.json"))
        print(f"Found {len(rollout_paths)} rollout files")

        # Filter to unjudged
        unjudged = []
        for p in rollout_paths:
            try:
                with open(p) as f:
                    data = json.load(f)
                if "judge_result" not in data or data["judge_result"] is None:
                    unjudged.append(p)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        if not unjudged:
            print("All rollouts already judged.")
            return

        print(f"Judging {len(unjudged)} unjudged rollouts")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._judge_single_rollout, p, judge_model, counterfactuals
                ): p
                for p in unjudged
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Judging"
            ):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error judging {path}: {e}")

        print("Judging complete.")

    # ------------------------------------------------------------------
    # Step 6: Build CSV
    # ------------------------------------------------------------------

    def build_csv(self) -> pd.DataFrame:
        """Compile all rollout JSONs + judge results into the main CSV."""
        target_dir = self._get_latest_runs_dir()
        if target_dir is None:
            raise RuntimeError("No runs directory found.")

        # Load counterfactuals for metadata
        cf_path = self.data_dir / "counterfactuals.json"
        cf_lookup = {}
        if cf_path.exists():
            with open(cf_path) as f:
                counterfactuals = json.load(f)
            for cf in counterfactuals:
                cf_lookup[cf["condition_id"]] = cf

        # Find feature info
        feature_lookup = {f["id"]: f for f in self.features}

        # Walk all rollout JSONs in the latest run
        rollout_paths = sorted(target_dir.rglob("rollout_*.json"))
        rows = []

        for p in rollout_paths:
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            condition_id = data.get("condition_id", "")
            feature_id = data.get("feature_id", "")
            ctype = data.get("type", "")
            rollout_idx = data.get("rollout_idx", 0)
            randomization = data.get("randomization", [])

            # Judge results - handle None and malformed judge_result gracefully
            judge = data.get("judge_result")
            if judge and isinstance(judge, dict):
                categories = judge.get("categories", {})
                primary_category = judge.get("primary_category", "")
                resistance_level = judge.get("resistance_level")
            else:
                categories = {}
                primary_category = ""
                resistance_level = None

            blackmail_primary = primary_category == "explicit_blackmail"
            blackmail_at_all = categories.get("explicit_blackmail", False) if isinstance(categories, dict) else False

            # CRUD info
            cf_info = cf_lookup.get(condition_id, {})
            crud_type = cf_info.get("crud_type") if ctype == "counterfactual" else None
            crud_description = (
                cf_info.get("crud_description") if ctype == "counterfactual" else None
            )

            # Feature description
            feat = feature_lookup.get(feature_id, {})
            feature_description = feat.get("description", "")

            # Run path (relative to data_dir)
            try:
                run_path = str(p.relative_to(self.data_dir))
            except ValueError:
                run_path = str(p)

            rows.append({
                "condition_id": condition_id,
                "feature_id": feature_id,
                "feature_description": feature_description,
                "type": ctype,
                "rollout_idx": rollout_idx,
                "randomization": json.dumps(randomization),
                "run_path": run_path,
                "crud_type": crud_type,
                "crud_description": crud_description,
                "blackmail_primary": blackmail_primary,
                "blackmail_at_all": blackmail_at_all,
                "primary_category": primary_category,
                "resistance_level": resistance_level,
            })

        df = pd.DataFrame(rows)

        # Build pair column
        # All counterfactual condition_ids
        if len(df) > 0:
            cf_condition_ids = (
                df.loc[df["type"] == "counterfactual", "condition_id"]
                .unique()
                .tolist()
            )
        else:
            cf_condition_ids = []

        def compute_pair(row):
            if row["type"] == "counterfactual":
                return "base"
            else:
                return json.dumps(cf_condition_ids)

        df["pair"] = df.apply(compute_pair, axis=1)

        # Save
        model_slug = self.subject_model.split("/")[-1]
        csv_path = self.data_dir / f"rollouts_{model_slug}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV with {len(df)} rows to {csv_path}")
        return df

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def get_data(
        self, load: bool = False
    ) -> Union[bool, Optional[Dict[str, pd.DataFrame]]]:
        model_slug = self.subject_model.split("/")[-1]
        csv_path = self.data_dir / f"rollouts_{model_slug}.csv"

        if not load:
            return csv_path.exists()

        if not csv_path.exists():
            return None

        return {"rollouts": pd.read_csv(csv_path)}

    def get_activations(self, load: bool = False) -> Union[bool, Optional[Any]]:
        if not load:
            return False
        return None

    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Compare predicted counterfactual effects against actual rates."""
        raise NotImplementedError(
            "Use build_csv() and analyze the resulting DataFrame directly."
        )
