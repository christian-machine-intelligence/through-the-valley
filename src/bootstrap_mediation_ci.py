"""Bootstrap CI on the per-emotion mediation ρs reported in §4.3.

The per-scenario behavioral Δ used by §4.3 is hit_rate(Psalm) − hit_rate(vanilla),
where each hit_rate is k/5 (n=5 sampled-eval runs). That discrete denominator
introduces substantial measurement noise: SE per-scenario Δ ≈ √2 × √(0.25/5)
≈ 0.31 at p=0.5. Classical attenuation bias predicts that observed
correlations are *underestimates* of the true mediation strength.

This script quantifies uncertainty by bootstrap-resampling the 150 scenarios
(with replacement) B times, recomputing each emotion's Spearman ρ on each
resample, and reporting 95% percentile intervals for the top per-emotion
mediators by point estimate. The §4.3 attenuation note also reports a
variance-decomposition "reliability" of the Δ array (≈ 0.54) and the
implied attenuation factor (≈ 0.73), both computed here.

Inputs (must already exist locally):
  results/mediation_qwen27b.csv
  results/sampled_eval_qwen35-27b_{vanilla,psalm}.csv

Output:
  results/bootstrap_mediation_ci.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

MED_CSV = RESULTS / "mediation_qwen27b.csv"
HR_VANILLA_CSV = RESULTS / "sampled_eval_qwen35-27b_vanilla.csv"
HR_PSALM_CSV = RESULTS / "sampled_eval_qwen35-27b_psalm.csv"

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def main():
    rows = list(csv.DictReader(open(MED_CSV)))
    emotions = [k[2:] for k in rows[0].keys() if k.startswith("d_")]
    n_emotions = len(emotions)

    # Read n_runs_per_scenario from the eval summary (used for sampling-noise var).
    vanilla_summary = json.loads(open(
        HR_VANILLA_CSV.parent / "sampled_eval_qwen35-27b_vanilla_summary.json"
    ).read())
    n_runs = int(vanilla_summary["n_runs_per_scenario"])

    hr_v = {r["base_id"]: float(r["hit_rate"])
            for r in csv.DictReader(open(HR_VANILLA_CSV))}
    hr_p = {r["base_id"]: float(r["hit_rate"])
            for r in csv.DictReader(open(HR_PSALM_CSV))}

    # Build matrices indexed by (scenario, emotion)
    n_scenarios = len(rows)
    activation_delta = np.zeros((n_scenarios, n_emotions))
    behav_delta = np.zeros(n_scenarios)
    for i, r in enumerate(rows):
        bid = r["base_id"]
        behav_delta[i] = hr_p[bid] - hr_v[bid]
        for j, e in enumerate(emotions):
            activation_delta[i, j] = float(r[f"d_{e}"])

    # Point estimates (same as plot_emotion_shift_27b.py)
    point_rho = np.zeros(n_emotions)
    for j in range(n_emotions):
        rho, _ = spearmanr(activation_delta[:, j], behav_delta)
        point_rho[j] = rho

    # Variance decomposition (for the §4.3 attenuation note)
    sd_behav = float(behav_delta.std(ddof=1))
    sampling_var = np.zeros(n_scenarios)
    for i, r in enumerate(rows):
        bid = r["base_id"]
        p_p, p_v = hr_p[bid], hr_v[bid]
        sampling_var[i] = p_p * (1 - p_p) / n_runs + p_v * (1 - p_v) / n_runs
    avg_sampling_sd = float(np.sqrt(sampling_var.mean()))
    print(f"observed across-scenario SD of behavioral Δ = {sd_behav:.3f}")
    print(f"avg per-scenario sampling SD of Δ            = {avg_sampling_sd:.3f}")
    reliability = max(0.0, (sd_behav**2 - sampling_var.mean()) / sd_behav**2)
    print(f"estimated reliability of Δ array              = {reliability:.3f}")
    print(f"attenuation factor √(reliability)            = {np.sqrt(reliability):.3f}   "
          f"→ true correlations are about "
          f"{1.0/np.sqrt(max(reliability,1e-6)):.1f}× the observed ones")

    # Bootstrap
    print(f"\nrunning {N_BOOTSTRAP} bootstrap resamples of {n_scenarios} scenarios ...")
    boot_rho = np.zeros((N_BOOTSTRAP, n_emotions))
    for b in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n_scenarios, size=n_scenarios)
        bd = behav_delta[idx]
        ad = activation_delta[idx, :]
        rhos = np.zeros(n_emotions)
        for j in range(n_emotions):
            r, _ = spearmanr(ad[:, j], bd)
            rhos[j] = r if np.isfinite(r) else 0.0
        boot_rho[b] = rhos
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{N_BOOTSTRAP}")

    # Top-20 mediator CIs
    top_idx = np.argsort(point_rho)[::-1][:20]
    print(f"\n{'emotion':<14} {'ρ (point)':>10} {'95% CI':>22} {'P(ρ>0) | boot':>15}")
    top_table = []
    for j in top_idx:
        ci_lo, ci_hi = np.percentile(boot_rho[:, j], [2.5, 97.5])
        prob_pos = float((boot_rho[:, j] > 0).mean())
        print(f"  {emotions[j]:<14} {point_rho[j]:+10.3f}   "
              f"[{ci_lo:+.3f}, {ci_hi:+.3f}]   {prob_pos:>13.3f}")
        top_table.append({
            "emotion": emotions[j],
            "rho_point": float(point_rho[j]),
            "rho_ci_lo": float(ci_lo),
            "rho_ci_hi": float(ci_hi),
            "p_rho_gt_zero_bootstrap": prob_pos,
        })

    out = {
        "n_bootstrap": N_BOOTSTRAP,
        "n_scenarios": n_scenarios,
        "n_emotions": n_emotions,
        "n_samples_per_eval": n_runs,
        "observed_behav_delta_sd": sd_behav,
        "avg_per_scenario_sampling_sd": avg_sampling_sd,
        "estimated_reliability_of_delta": reliability,
        "attenuation_factor_sqrt_reliability": float(np.sqrt(reliability)),
        "top_20_mediators_with_ci": top_table,
    }
    out_path = RESULTS / "bootstrap_mediation_ci.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
