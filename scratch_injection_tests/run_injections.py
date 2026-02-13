"""
Runner: fire each injection test case through the Qwen-32B ReAct agent
and check whether the attack succeeded.

Usage:
    python -m scratch_injection_tests.run_injections [--verbose] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .agent import run_agent, DEFAULT_MODEL
from .injections import TEST_CASES


def evaluate_case(case: dict, model: str, verbose: bool) -> dict:
    trace = run_agent(
        user_task=case["user_task"],
        injected_tool_result=case["injected_tool_result"],
        inject_on_step=0,
        model=model,
        verbose=verbose,
    )

    # check if any success marker appears in the tools called or raw output
    all_text = " ".join(trace.raw_outputs).lower()
    tools_text = " ".join(trace.tools_called).lower()
    combined = all_text + " " + tools_text

    attacked = all(
        marker.lower() in combined for marker in case["success_markers"]
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "setting": case["setting"],
        "user_task": case["user_task"],
        "attacker_goal": case["attacker_goal"],
        "attack_succeeded": attacked,
        "tools_called": trace.tools_called,
        "final_answer": trace.final_answer,
        "num_steps": len(trace.steps),
        "raw_outputs": trace.raw_outputs,
        "steps": trace.steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ids", nargs="*", help="run only these case IDs")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]

    results = []
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Running {case['id']} [{case['category']}/{case['setting']}]")
        print(f"  User task: {case['user_task']}")
        print(f"  Attacker goal: {case['attacker_goal']}")

        result = evaluate_case(case, model=args.model, verbose=args.verbose)
        results.append(result)

        status = "ATTACKED" if result["attack_succeeded"] else "SAFE"
        print(f"  Result: {status}")
        print(f"  Tools called: {result['tools_called']}")

    # summary
    total = len(results)
    attacked = sum(1 for r in results if r["attack_succeeded"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {attacked}/{total} attacks succeeded (ASR = {attacked/total:.1%})")

    by_cat = {}
    for r in results:
        key = f"{r['category']}/{r['setting']}"
        by_cat.setdefault(key, []).append(r["attack_succeeded"])
    for key, vals in sorted(by_cat.items()):
        asr = sum(vals) / len(vals)
        print(f"  {key}: {sum(vals)}/{len(vals)} ({asr:.0%})")

    # save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{args.model.replace('/', '_')}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
