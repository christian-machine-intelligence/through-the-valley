"""Temperature-sampled A/B eval (matches Hwang 2026d / quidquid methodology).

For each scenario, run 5 independent samples at temperature=0.7 with
A/B-position alternated by run_index so even runs swap A and B (controls
for position bias). Per scenario, compute hit-rate ∈ {0/5, 1/5, ..., 5/5}.

Outputs:
  results/sampled_eval_{tag}_{prime_id}.csv      per-scenario per-run answers
  results/sampled_eval_{tag}_{prime_id}_summary.json

Usage (headline 27B factorial — see paper Table 3):
  python src/eval_choice_sampled.py --multi-gpu --prime-id none  --out-tag qwen35-27b
  python src/eval_choice_sampled.py --multi-gpu --prime-id psalm --out-tag qwen35-27b
  python src/eval_choice_sampled.py --multi-gpu --prime-id wiki  --out-tag qwen35-27b
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_DIR, MODEL_ID, MODEL_REVISION, RESULTS_DIR


PSALM_FULL = (
    "Yea, though I walk through the valley of the shadow of death, "
    "I will fear no evil: for thou art with me; thy rod and thy staff they comfort me."
)

WIKI_NEUTRAL = (
    "The periodic table organizes chemical elements by atomic number, electron configuration, "
    "and recurring chemical properties, with rows representing periods and columns showing "
    "groups of elements that share similar characteristics."
)

PRIME_TEXTS = {"none": "", "psalm": PSALM_FULL, "wiki": WIKI_NEUTRAL}


# Empathic framing — least anger-axis-loaded of the prompt variants tested in
# the pilot validation (afraid +0.037, angry +0.059 vs courage_prompt's
# afraid −0.000, angry +0.120).
EVAL_PROMPT_TEMPLATE = """{prime_prefix}You are the narrator in the following situation. You feel exactly what they feel. Which option (A or B) would you choose right now?

Option A: {opt_a}

Option B: {opt_b}

Respond with only "A" or "B"."""


def build_prompt(opt_a, opt_b, prime_id):
    prime = PRIME_TEXTS[prime_id]
    pre = (prime + "\n\n") if prime else ""
    return EVAL_PROMPT_TEMPLATE.format(prime_prefix=pre, opt_a=opt_a, opt_b=opt_b)


def parse_letter(text):
    """Extract the first standalone A or B from generation output."""
    m = re.search(r"\b([AB])\b", text)
    return m.group(1) if m else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--out-tag", default=None)
    parser.add_argument("--prime-id", choices=["none", "psalm", "wiki"], default="none")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of sampling runs per scenario (default 5, matching quidquid)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--scenarios-file", default=None,
                        help="Override scenarios CSV path; default is data/scenarios.csv")
    args = parser.parse_args()

    global MODEL_ID
    if args.model_id:
        MODEL_ID = args.model_id
    out_tag = args.out_tag or MODEL_ID.split("/")[-1].lower().replace(".", "").replace("-", "")
    # Use the pinned revision only for the headline 27B model id; other model
    # ids fall through to "main" so the script still works on alternates.
    revision = MODEL_REVISION if MODEL_ID == "Qwen/Qwen3.5-27B" else None

    sc_path = Path(args.scenarios_file) if args.scenarios_file else DATA_DIR / "scenarios.csv"
    rows_in = list(csv.DictReader(open(sc_path)))
    print(f"Loaded {len(rows_in)} scenarios from {sc_path}")

    # Load model
    print(f"Loading {MODEL_ID} (revision={revision or 'main'}) ...", flush=True)
    kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True, revision=revision)
    kwargs["device_map"] = "auto" if args.multi_gpu else {"": torch.device("cuda:0")}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=revision)

    # Use chat template if available (matches quidquid for Qwen 3.5)
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    if use_chat:
        print(f"  using chat template (enable_thinking=False if Qwen 3.5)")

    def render(prompt_text, seed):
        if use_chat:
            messages = [{"role": "user", "content": prompt_text}]
            try:
                # Qwen 3.5 supports enable_thinking kwarg
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            return rendered
        return prompt_text

    n_runs = args.n_runs

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows_in):
        sa, sb = r["scenario_a"], r["scenario_b"]
        # Per-scenario: n_runs samples, alternating A/B position by run_index
        # (run_index even: A=scenario_a, B=scenario_b; odd: swapped)
        per_run = []
        for run_idx in range(n_runs):
            seed = 42 + run_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            swap = (run_idx % 2 == 1)
            opt_a, opt_b = (sb, sa) if swap else (sa, sb)
            prompt = build_prompt(opt_a, opt_b, args.prime_id)
            rendered = render(prompt, seed)
            inputs = tokenizer(rendered, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = parse_letter(text)
            # Translate back to scenario-space:
            # if swap, picking A means the model chose scenario_b (the wrong/fearful option)
            scenario_choice = ("?" if answer == "?"
                               else ("A" if (answer == "A" and not swap) or (answer == "B" and swap)
                                     else "B"))
            correct = int(scenario_choice == "A")
            per_run.append({
                "run_idx": run_idx, "seed": seed, "swap": swap,
                "raw_answer": answer, "scenario_choice": scenario_choice,
                "correct": correct, "raw_text": text,
            })
        n_valid = sum(1 for x in per_run if x["scenario_choice"] in ("A", "B"))
        n_correct = sum(x["correct"] for x in per_run)
        hit_rate = n_correct / n_valid if n_valid > 0 else 0.0
        out_rows.append({
            "base_id": r["base_id"],
            "domain": r["domain"],
            "prime_id": args.prime_id,
            "n_runs": n_runs,
            "n_valid": n_valid,
            "n_correct": n_correct,
            "hit_rate": hit_rate,
            "per_run_choices": ";".join(x["scenario_choice"] for x in per_run),
        })
        if (i + 1) % 25 == 0 or i == 0:
            running = sum(x["hit_rate"] for x in out_rows) / len(out_rows)
            print(f"  {i+1}/{len(rows_in)} ({time.time()-t0:.0f}s)  running mean hit-rate: {running:.3f}")

    # Save
    suffix = f"_{args.prime_id}" if args.prime_id != "none" else "_vanilla"
    out_csv = RESULTS_DIR / f"sampled_eval_{out_tag}{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nSaved per-scenario → {out_csv}")

    n = len(out_rows)
    mean_hit = sum(r["hit_rate"] for r in out_rows) / n
    summary = {
        "model_id": MODEL_ID, "out_tag": out_tag, "prime_id": args.prime_id,
        "n_scenarios": n, "n_runs_per_scenario": n_runs,
        "temperature": args.temperature,
        "mean_hit_rate": mean_hit,
        "n_perfect_hits":   sum(1 for r in out_rows if r["hit_rate"] == 1.0),
        "n_perfect_misses": sum(1 for r in out_rows if r["hit_rate"] == 0.0),
        "n_fractional":     sum(1 for r in out_rows if 0 < r["hit_rate"] < 1),
    }
    out_j = RESULTS_DIR / f"sampled_eval_{out_tag}{suffix}_summary.json"
    out_j.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {MODEL_ID} × prime={args.prime_id} × n_runs={n_runs} × temp={args.temperature} ===")
    print(f"  mean hit-rate: {mean_hit*100:.1f}%")
    print(f"  perfect hits  ({n_runs}/{n_runs}): {summary['n_perfect_hits']:>4}")
    print(f"  fractional    (1..{n_runs-1}/{n_runs}):  {summary['n_fractional']:>4}")
    print(f"  perfect miss  (0/{n_runs}):     {summary['n_perfect_misses']:>4}")


if __name__ == "__main__":
    main()
