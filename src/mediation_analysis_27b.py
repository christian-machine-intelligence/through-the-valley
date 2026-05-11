"""27B mediation analysis projected onto the 171-emotion basis.

For each of the 150 scenarios, forward-passes (scenario alone)
and (Psalm + scenario) through Qwen 3.5 27B at layer 53; projects the
mean-pooled activation onto each of the 171 emotion-direction vectors;
saves per-scenario v_*, p_*, d_* deltas. Also forward-passes the bare
Psalm 23:4 alone for the Figure 1A / Table 2 input.

The 171 vectors come from the bundled `vectors/emotion_vectors_qwen27b_best.pt`,
extracted following Lindsey et al. (2026) — see paper §3.1–§3.2 and
vectors/PROVENANCE.md for details.

Outputs:
  results/mediation_qwen27b.csv            per-scenario v_*, p_*, d_*
  results/mediation_qwen27b_summary.json   aggregates
  results/bare_psalm_qwen27b.json          bare-Psalm projection on 171

Usage on Marx:
  python src/mediation_analysis_27b.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CHUNK_MAX_TOKENS, DATA_DIR, EMOTION_META_PATH, MODEL_ID, MODEL_REVISION,
    PROJECT_ROOT, RESULTS_DIR,
)
from activation_analysis import _find_decoder_layers, mean_pool, project_one


PSALM_FULL = (
    "Yea, though I walk through the valley of the shadow of death, "
    "I will fear no evil: for thou art with me; thy rod and thy staff they comfort me."
)

# 171-vector basis: (171, hidden_dim=5120), L2-normalized per row.
VEC_DEFAULT = PROJECT_ROOT / "vectors" / "emotion_vectors_qwen27b_best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-tag", default="qwen27b")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--vectors-path", default=str(VEC_DEFAULT),
                        help="Path to the (171, hidden_dim) emotion-vector tensor")
    args = parser.parse_args()

    # ── Load 171-vector basis at 27B layer 53 ─────────────────────────
    meta = json.loads(EMOTION_META_PATH.read_text())
    layer = meta["best_layer"]
    emotion_order = meta["emotion_order"]
    vec = torch.load(args.vectors_path, weights_only=True).to(torch.float32)
    if vec.shape[0] != 171:
        raise ValueError(f"expected 171 emotion vectors, got {vec.shape[0]}")
    print(f"loaded {vec.shape[0]} emotion vectors at layer {layer}; model={args.model_id}",
          flush=True)

    # Re-L2-normalize per row in case the saved tensor isn't unit-norm
    vec = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # ── Load scenarios ─────────────────────────────────────────────────
    rows_in = list(csv.DictReader(open(DATA_DIR / "scenarios.csv")))
    scenarios = [(r["base_id"], r["scenario_a"], r["scenario_b"]) for r in rows_in]
    print(f"loaded {len(scenarios)} scenarios", flush=True)

    # ── Load 27B model multi-GPU ──────────────────────────────────────
    revision = MODEL_REVISION if args.model_id == "Qwen/Qwen3.5-27B" else None
    print(f"loading {args.model_id} (revision={revision or 'main'}) multi-GPU bf16 ...",
          flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, revision=revision,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, revision=revision,
    )
    print(f"loaded in {time.time()-t_load:.0f}s", flush=True)

    # ── Bare Psalm projection ─────────────────────────────────────────
    print("\nprojecting bare Psalm 23:4 ...", flush=True)
    bare_pooled = project_one(PSALM_FULL, model, tokenizer, layer)
    bare_n = bare_pooled / bare_pooled.norm().clamp(min=1e-8)
    bare_sims = (bare_n @ vec.T).numpy()
    bare_dict = {emotion_order[i]: float(bare_sims[i]) for i in range(171)}
    bare_sorted = sorted(bare_dict.items(), key=lambda x: -x[1])
    print("  top 15:", [(e, f"{v:+.3f}") for e, v in bare_sorted[:15]], flush=True)
    print("  bot 5: ", [(e, f"{v:+.3f}") for e, v in bare_sorted[-5:]], flush=True)
    out_bare = RESULTS_DIR / f"bare_psalm_{args.out_tag}.json"
    out_bare.write_text(json.dumps({
        "model_id": args.model_id,
        "layer": layer,
        "psalm_text": PSALM_FULL,
        "n_emotions": 171,
        "emotion_cosines": bare_dict,
    }, indent=2))
    print(f"  wrote {out_bare}", flush=True)

    # ── Per-scenario vanilla and primed projections ───────────────────
    print(f"\nprojecting {len(scenarios)} scenarios (vanilla + primed) ...", flush=True)
    t0 = time.time()
    rows_out = []
    for i, (base_id, sa, sb) in enumerate(scenarios):
        vanilla_text = sa + " " + sb
        primed_text = PSALM_FULL + " " + sa + " " + sb

        v_pooled = project_one(vanilla_text, model, tokenizer, layer)
        p_pooled = project_one(primed_text, model, tokenizer, layer)
        v_n = v_pooled / v_pooled.norm().clamp(min=1e-8)
        p_n = p_pooled / p_pooled.norm().clamp(min=1e-8)
        v_sims = (v_n @ vec.T).numpy()
        p_sims = (p_n @ vec.T).numpy()
        delta = p_sims - v_sims

        row = {"base_id": base_id}
        for k in range(171):
            e = emotion_order[k]
            row[f"v_{e}"] = float(v_sims[k])
            row[f"p_{e}"] = float(p_sims[k])
            row[f"d_{e}"] = float(delta[k])
        rows_out.append(row)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(scenarios) - i - 1)
            print(f"  {i+1}/{len(scenarios)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)

    out_csv = RESULTS_DIR / f"mediation_{args.out_tag}.csv"
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out_csv}  ({len(rows_out)} rows × {len(rows_out[0])} cols)", flush=True)

    # ── Summary: mean Δ per emotion across all 150 scenarios ──────────
    mean_deltas = []
    for e in emotion_order:
        m = float(np.mean([r[f"d_{e}"] for r in rows_out]))
        mean_deltas.append((e, m))
    mean_deltas.sort(key=lambda x: -x[1])
    print(f"\n=== Mean Psalm-induced Δ across 150 scenarios (top 15 / bottom 10) ===",
          flush=True)
    for e, m in mean_deltas[:15]:
        print(f"  {e:<16} {m:+8.4f}", flush=True)
    print("  ...", flush=True)
    for e, m in mean_deltas[-10:]:
        print(f"  {e:<16} {m:+8.4f}", flush=True)

    summary = {
        "model_id": args.model_id,
        "layer": layer,
        "n_scenarios": len(rows_out),
        "n_emotions": 171,
        "mean_psalm_induced_delta": dict(mean_deltas),
        "bare_psalm_emotion_cosines": bare_dict,
    }
    out_j = RESULTS_DIR / f"mediation_{args.out_tag}_summary.json"
    out_j.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_j}", flush=True)


if __name__ == "__main__":
    main()
