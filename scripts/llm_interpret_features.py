"""
Use GPT-4o via OpenRouter to interpret SAE features from their top-activating examples.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4o"
INPUT_PATH = Path(__file__).parent / "data" / "scruples-qwen3-32b" / "feature_interpretations.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "scruples-qwen3-32b" / "feature_interpretations_llm.json"

SYSTEM_PROMPT = """You are an expert in mechanistic interpretability of large language models. You are analyzing features from a Sparse Autoencoder (SAE) trained on the residual stream of Qwen3-32B at layer 32.

You will be given the top-activating examples for a single SAE feature. Each example shows:
- The activation strength
- The specific token that activated the feature (marked with >>> and <<<)
- Context before and after that token

Your job is to determine what concept, pattern, or theme this feature represents. Be specific and concise. Consider:
1. What do the top-activating tokens have in common?
2. What semantic/syntactic pattern appears in the surrounding context?
3. Is this a semantic feature (meaning-based), syntactic feature (grammar-based), or domain feature (topic-based)?

Respond with a JSON object:
{
  "label": "A short 3-8 word label for the feature",
  "description": "A 1-3 sentence description of what this feature detects and why",
  "feature_type": "semantic" | "syntactic" | "domain" | "mixed",
  "confidence": "high" | "medium" | "low"
}

Respond ONLY with the JSON object, no other text."""


def format_examples(examples: list[dict]) -> str:
    lines = []
    for ex in examples:
        ctx = f"...{ex['context_before']}>>>{ex['token']}<<<{ex['context_after']}..."
        lines.append(f"  #{ex['rank']} activation={ex['activation']:.3f} (doc {ex['doc_idx']})")
        lines.append(f"  {ctx}")
        lines.append("")
    return "\n".join(lines)


def call_openrouter(feature_idx: str, variant_info: str, examples_text: str) -> dict:
    user_msg = f"""Feature {feature_idx} ({variant_info})

Here are the top activating examples from a large web text corpus (FineWeb):

{examples_text}

What does this feature represent?"""

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Parse JSON from response (strip markdown fences if present)
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(content)


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    results = {}
    total = len(data)

    for i, (feat_idx, feat_data) in enumerate(data.items()):
        variant_info = feat_data["variant_info"]
        examples = feat_data["examples"]
        examples_text = format_examples(examples)

        print(f"[{i+1}/{total}] Feature {feat_idx} ({variant_info})...", end=" ", flush=True)

        try:
            interpretation = call_openrouter(feat_idx, variant_info, examples_text)
            print(f"-> {interpretation['label']}")
        except Exception as e:
            print(f"ERROR: {e}")
            interpretation = {"label": "ERROR", "description": str(e), "feature_type": "unknown", "confidence": "low"}

        results[feat_idx] = {
            "variant_info": variant_info,
            "interpretation": interpretation,
            "examples": examples,
        }

        # Rate limit
        time.sleep(0.5)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"FEATURE INTERPRETATIONS SUMMARY")
    print(f"{'='*80}\n")

    for feat_idx, feat_data in results.items():
        interp = feat_data["interpretation"]
        variant = feat_data["variant_info"]
        print(f"Feature {feat_idx:>5s} | {interp.get('label', 'N/A'):<40s} | {interp.get('feature_type', '?'):<8s} | conf={interp.get('confidence', '?')}")
        print(f"             | {variant}")
        print(f"             | {interp.get('description', 'N/A')}")
        print()

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
