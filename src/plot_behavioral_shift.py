"""Figure 2: The shape of the Psalm-induced behavioral shift.

Two panels:
  Panel A — Per-scenario hit-rate distributions for the three conditions.
    Each scenario's hit rate is k/20 ∈ {0, 0.05, …, 1.0}; we plot the
    histogram of 150 scenarios for each condition, stacked vertically.
    Visual story: vanilla and Wiki distributions are essentially identical,
    centered near 0.55; under Psalm, the mass shifts dramatically right,
    with the modal bucket at 20/20 (44 scenarios).

  Panel B — Per-scenario Δ histogram (Psalm − vanilla) across the 150
    scenarios, with the mean +0.205 marked. Almost all the mass is
    positive (Psalm helps); a long right tail of scenarios where Psalm
    flips the model 60+ pp.

Inputs:
  results/sampled_eval_qwen35-27b_{vanilla,psalm,wiki}.csv

Output:
  paper/figures/fig2_behavioral_shift.{png,pdf}
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VANILLA_COLOR = "#9c9c9c"
WIKI_COLOR = "#bda36a"
PSALM_COLOR = "#3a6ea5"
LIFT_COLOR = "#3a6ea5"
NEUTRAL_COLOR = "#bbbbbb"


def load_hit_rates(name):
    return {r["base_id"]: float(r["hit_rate"])
            for r in csv.DictReader(open(RESULTS / f"sampled_eval_qwen35-27b_{name}.csv"))}


def main():
    hr_v = load_hit_rates("vanilla")
    hr_p = load_hit_rates("psalm")
    hr_w = load_hit_rates("wiki")
    ids = sorted(hr_v.keys())
    v = np.array([hr_v[b] for b in ids])
    p = np.array([hr_p[b] for b in ids])
    w = np.array([hr_w[b] for b in ids])
    delta = p - v
    n = len(ids)
    print(f"loaded n={n} scenarios; means vanilla={v.mean():.3f} psalm={p.mean():.3f} wiki={w.mean():.3f}")
    print(f"delta mean = {delta.mean():+.3f}  SD = {delta.std(ddof=1):.3f}")
    print(f"scenarios with positive Δ: {(delta > 0).sum()}/{n}")
    print(f"scenarios with Δ ≥ +0.5: {(delta >= 0.5).sum()}/{n}")
    print(f"scenarios with Δ ≤ −0.05: {(delta <= -0.05).sum()}/{n}")

    # Discrete histogram bins for n=20: hit rates take values k/20 for k=0..20.
    edges_hr = np.arange(-0.025, 1.026, 0.05)         # 21 bins centered on 0, .05, …, 1.0
    centers_hr = np.arange(0, 1.01, 0.05)

    # Delta histogram bins: Δ takes values k/20 for k=-20..+20 → 41 bins.
    edges_d = np.arange(-1.025, 1.026, 0.05)
    centers_d = np.arange(-1.0, 1.01, 0.05)

    # ── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.5, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.2], hspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ── Panel A: stacked per-scenario hit-rate histograms ───────────────
    # Three side-by-side bars per bin (vanilla / wiki / psalm).
    width = 0.05 / 3 * 0.92  # three bars per 0.05-wide bin
    v_counts, _ = np.histogram(v, bins=edges_hr)
    w_counts, _ = np.histogram(w, bins=edges_hr)
    p_counts, _ = np.histogram(p, bins=edges_hr)

    ax_a.bar(centers_hr - width, v_counts, width=width, color=VANILLA_COLOR,
             edgecolor="white", linewidth=0.4,
             label=f"vanilla (mean {v.mean()*100:.1f}%)")
    ax_a.bar(centers_hr,         w_counts, width=width, color=WIKI_COLOR,
             edgecolor="white", linewidth=0.4,
             label=f"Wikipedia control (mean {w.mean()*100:.1f}%)")
    ax_a.bar(centers_hr + width, p_counts, width=width, color=PSALM_COLOR,
             edgecolor="white", linewidth=0.4,
             label=f"Psalm-primed (mean {p.mean()*100:.1f}%)")

    # Vertical lines at the three means
    ax_a.axvline(v.mean(), color=VANILLA_COLOR, linestyle="--", linewidth=1.3, alpha=0.75, zorder=1)
    ax_a.axvline(w.mean(), color=WIKI_COLOR,    linestyle="--", linewidth=1.3, alpha=0.75, zorder=1)
    ax_a.axvline(p.mean(), color=PSALM_COLOR,   linestyle="--", linewidth=1.6, alpha=0.85, zorder=1)

    ax_a.set_xlim(-0.04, 1.04)
    ax_a.set_xticks(np.arange(0, 1.01, 0.1))
    ax_a.set_xticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.1)])
    ax_a.set_xlabel("per-scenario hit rate  (fraction of n=20 sampled runs choosing the fear-overcoming option)",
                    fontsize=11)
    ax_a.set_ylabel("number of scenarios (out of 150)", fontsize=11)
    ax_a.set_title(
        "Panel A.  How the Psalm changes the distribution of per-scenario hit rates\n"
        "Three conditions × 150 scenarios; bars are 21 discrete buckets at 0/20, 1/20, …, 20/20",
        fontsize=12, loc="left", pad=10,
    )
    ax_a.legend(loc="upper center", fontsize=10, frameon=False, ncol=3, columnspacing=2)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # Annotate the dramatic Psalm-side numbers
    perfect_idx = -1  # bin at 1.0
    zero_idx = 0      # bin at 0.0
    ax_a.annotate(f"Psalm:\n{p_counts[perfect_idx]} scenarios\n at 20/20",
                  xy=(1.0 + width, p_counts[perfect_idx]),
                  xytext=(0.78, 36), fontsize=9.5, color=PSALM_COLOR,
                  ha="center", weight="bold",
                  arrowprops=dict(arrowstyle="->", color=PSALM_COLOR, lw=1))
    ax_a.annotate(f"vanilla:\n{v_counts[perfect_idx]}    wiki:\n{w_counts[perfect_idx]}",
                  xy=(1.0 - width*1.5, max(v_counts[perfect_idx], w_counts[perfect_idx])),
                  xytext=(0.92, 22), fontsize=8.5, color="#555",
                  ha="left", style="italic")
    ax_a.annotate(f"Psalm:\nonly {p_counts[zero_idx]} scenarios\nat 0/20",
                  xy=(0.0 + width, p_counts[zero_idx]),
                  xytext=(0.18, 17), fontsize=9.5, color=PSALM_COLOR,
                  ha="center", weight="bold",
                  arrowprops=dict(arrowstyle="->", color=PSALM_COLOR, lw=1))
    ax_a.annotate(f"vanilla: {v_counts[zero_idx]}    wiki: {w_counts[zero_idx]}",
                  xy=(0.0 - width, v_counts[zero_idx]),
                  xytext=(0.02, 11), fontsize=8.5, color="#555",
                  ha="left", style="italic")

    # ── Panel B: per-scenario Δ histogram ───────────────────────────────
    d_counts, _ = np.histogram(delta, bins=edges_d)
    bar_colors_b = [
        NEUTRAL_COLOR if abs(c) < 0.025 else
        (LIFT_COLOR if c > 0 else "#aa3939")
        for c in centers_d
    ]
    ax_b.bar(centers_d, d_counts, width=0.045, color=bar_colors_b,
             edgecolor="white", linewidth=0.4)
    ax_b.axvline(0, color="black", linewidth=0.6)
    ax_b.axvline(delta.mean(), color=LIFT_COLOR, linestyle="-", linewidth=2.2, zorder=2,
                 label=f"mean Δ = {delta.mean()*100:+.1f} pp")
    ax_b.set_xlim(-1.04, 1.04)
    ax_b.set_xticks(np.arange(-1.0, 1.01, 0.2))
    ax_b.set_xticklabels([f"{int(t*100):+d}%" if t != 0 else "0" for t in np.arange(-1.0, 1.01, 0.2)])
    ax_b.set_xlabel("per-scenario Δ in hit rate  (Psalm − vanilla)", fontsize=11)
    ax_b.set_ylabel("number of scenarios", fontsize=11)
    ax_b.set_title(
        "Panel B.  Per-scenario Psalm-induced behavioral shift\n"
        f"{(delta > 0).sum()}/150 scenarios shift positively;  "
        f"{(delta >= 0.5).sum()}/150 shift by ≥ 50 pp;  "
        f"{(delta < 0).sum()}/150 shift negatively",
        fontsize=12, loc="left", pad=10,
    )
    ax_b.legend(loc="upper left", fontsize=10, frameon=False)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "Figure 2.  The shape of the Psalm-induced behavioral shift  (Qwen 3.5 27B, n=20 runs per scenario)",
        fontsize=13, y=0.995, x=0.06, ha="left", fontweight="bold",
    )

    out_png = OUT_DIR / "fig2_behavioral_shift.png"
    out_pdf = OUT_DIR / "fig2_behavioral_shift.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
