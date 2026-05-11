"""Render Figures 1 and 3 on the 171-emotion basis at Qwen 3.5 27B layer 53,
and write the per-emotion mediation statistics (Spearman ρ, BH-FDR significance,
Bonferroni significance) used by §4.4 of the paper.

(Figure 2 — behavioral-shift distributions — is produced by
plot_behavioral_shift.py.)

Inputs (must already exist locally — pull from Marx after running
mediation_analysis_27b.py):
  results/mediation_qwen27b.csv          per-scenario v_*, p_*, d_*
  results/bare_psalm_qwen27b.json        bare-Psalm projection on 171
  results/sampled_eval_qwen35-27b_vanilla.csv   per-scenario vanilla hit_rate
  results/sampled_eval_qwen35-27b_psalm.csv     per-scenario psalm hit_rate

Figures:
  Figure 1. Bare Psalm 23:4 fingerprint — cosine of the bare Psalm against
            each emotion-direction vector. Top 20 + bottom 10 of 171, ranked
            by signed cosine.
  Figure 3. In-context Psalm-induced shift — mean per-emotion Δ across the
            150 scenarios. Top 15 + bottom 15 of 171, ranked by signed Δ.

Outputs:
  paper/figures/fig1_bare_psalm_fingerprint.{png,pdf}
  paper/figures/fig3_in_context_shift.{png,pdf}
  results/mediation_qwen27b_stats.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MED_CSV = RESULTS / "mediation_qwen27b.csv"
BARE_JSON = RESULTS / "bare_psalm_qwen27b.json"
HR_VANILLA_CSV = RESULTS / "sampled_eval_qwen35-27b_vanilla.csv"
HR_PSALM_CSV = RESULTS / "sampled_eval_qwen35-27b_psalm.csv"

POS_COLOR = "#3a6ea5"
NEG_COLOR = "#aa3939"
GRAY = "#bbbbbb"
LINE_RED = "#aa3939"

# Affective register groupings used in the paper (§4.4 + §4.3).
# Each emotion gets at most one register; emotions not in any register
# receive the default label color. The colors are intended to be
# distinguishable both on screen and in print.
REGISTER_COLORS = {
    "awe-courage":       "#1f4e79",  # deep blue — awe + resolute-courage register
    "penitential":       "#7a3a8a",  # purple — contrition / regret
    "compassion":        "#2e7a3e",  # warm green — care-for-other
    "moral-indignation": "#c0392b",  # dark red — thumos
    "hope-gratitude":    "#d4a017",  # gold — hope / gratitude
    "disordered":        "#888888",  # gray — anger / vexation / distress (the
                                     # register the Psalm consistently reduces)
}
REGISTER_MEMBERS = {
    "awe-courage": [
        "surprised", "valiant", "awestruck", "amazed", "bewildered", "mad",
        "defiant", "astonished", "shocked", "mystified", "perplexed",
        "obstinate", "grief-stricken", "heartbroken", "melancholy",
    ],
    "penitential": [
        "sorry", "remorseful", "ashamed", "regretful", "guilty",
        "self-critical", "humiliated",
    ],
    "compassion": [
        "compassionate", "kind", "empathetic", "loving", "patient", "docile",
        "sympathetic",
    ],
    "moral-indignation": [
        "bitter", "offended", "smug", "vindictive", "indignant", "insulted",
        "envious", "spiteful", "contemptuous", "suspicious", "scornful",
        "hateful", "disdainful", "jealous",
    ],
    "hope-gratitude": [
        "hope", "grateful", "jubilant", "hopeful", "thankful", "elated",
    ],
    "disordered": [
        "angry", "irritated", "enraged", "furious", "irate", "frustrated",
        "grumpy", "stressed", "upset", "overwhelmed", "dispirited",
        "distressed", "annoyed", "miserable", "restless", "impatient",
        "desperate", "tense", "exasperated", "resentful", "hostile",
    ],
}
EMO_TO_REGISTER = {e: r for r, es in REGISTER_MEMBERS.items() for e in es}


def register_color(emotion):
    """Color for the emotion's y-axis tick label, by affective register."""
    return REGISTER_COLORS.get(EMO_TO_REGISTER.get(emotion), "#222222")


def add_register_legend(fig, loc="upper right"):
    """Add a legend mapping color → affective register, anchored on the figure."""
    from matplotlib.patches import Patch
    handles = [
        Patch(color=REGISTER_COLORS["awe-courage"],       label="awe-courage"),
        Patch(color=REGISTER_COLORS["penitential"],       label="penitential"),
        Patch(color=REGISTER_COLORS["compassion"],        label="compassion"),
        Patch(color=REGISTER_COLORS["moral-indignation"], label="moral-indignation"),
        Patch(color=REGISTER_COLORS["hope-gratitude"],    label="hope-gratitude"),
        Patch(color=REGISTER_COLORS["disordered"],        label="disordered (anger/vexation/distress)"),
    ]
    fig.legend(
        handles=handles, loc=loc, fontsize=9, frameon=True,
        framealpha=0.92, title="affective register (label color)",
        title_fontsize=9.5, ncol=3, borderaxespad=0.8,
        bbox_to_anchor=(0.5, 0.01),
    )


def bh_fdr(p_values, q=0.05):
    """Benjamini-Hochberg step-up. Returns boolean mask of which p-values are
    significant at FDR q, plus the BH-adjusted critical p-threshold."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = np.arange(1, n + 1) / n * q
    sig_idx = np.where(ranked <= crit)[0]
    if len(sig_idx) == 0:
        return np.zeros(n, dtype=bool), 0.0
    k_max = sig_idx.max()
    threshold = ranked[k_max]
    return p <= threshold, float(threshold)


def save(fig, stem):
    """Save fig as PNG and PDF in OUT_DIR with the given filename stem."""
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def figure_1_bare_psalm(stats, emotions):
    """Bare-Psalm fingerprint: bar chart of top-20 + bottom-10 emotions by
    bare-Psalm cosine. Bar color = signed shift (blue positive, red negative).
    Y-axis label color = affective-register membership (see REGISTER_COLORS)."""
    by_bare = sorted(emotions, key=lambda e: stats[e]["bare_cosine"], reverse=True)
    selected = by_bare[:20] + by_bare[-10:]
    vals = np.array([stats[e]["bare_cosine"] for e in selected])
    y = np.arange(len(selected))
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(selected, fontsize=10.5)
    for tick, emo in zip(ax.get_yticklabels(), selected):
        tick.set_color(register_color(emo))
        if EMO_TO_REGISTER.get(emo):
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("cosine similarity (bare Psalm 23:4 → emotion vector)",
                  fontsize=11)
    ax.set_title(
        "Figure 1.  Bare Psalm 23:4 emotional fingerprint\n"
        "Qwen 3.5 27B, layer 53 — top 20 + bottom 10 of 171 emotion directions",
        fontsize=12, loc="left", pad=12,
    )
    for i, v in enumerate(vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9,
                color="#222")
    ax.spines[["top", "right"]].set_visible(False)
    span = max(abs(vals.min()), abs(vals.max())) * 1.25
    ax.set_xlim(-span, span)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    add_register_legend(fig, loc="lower center")
    save(fig, "fig1_bare_psalm_fingerprint")
    plt.close(fig)


def figure_2_in_context(stats, emotions, n_scenarios):
    """In-context shift: bar chart of top-15 + bottom-15 emotions by mean
    Psalm-induced activation Δ across the 150 scenarios. Bar color = signed
    shift; y-axis label color = affective-register membership."""
    by_delta = sorted(emotions, key=lambda e: stats[e]["mean_delta"], reverse=True)
    selected = by_delta[:15] + by_delta[-15:]
    vals = np.array([stats[e]["mean_delta"] for e in selected])
    y = np.arange(len(selected))
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(selected, fontsize=10.5)
    for tick, emo in zip(ax.get_yticklabels(), selected):
        tick.set_color(register_color(emo))
        if EMO_TO_REGISTER.get(emo):
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("mean Psalm-induced Δ across 150 scenarios\n"
                  "= cos(Psalm+scenario, emotion) − cos(scenario alone, emotion), averaged",
                  fontsize=11)
    ax.set_title(
        f"Figure 2.  In-context Psalm-induced activation shift\n"
        f"Qwen 3.5 27B, layer 53 — top 15 + bottom 15 of 171 emotion directions "
        f"(n = {n_scenarios} scenarios)",
        fontsize=12, loc="left", pad=12,
    )
    for i, v in enumerate(vals):
        ax.text(v + (0.0008 if v >= 0 else -0.0008), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9,
                color="#222")
    ax.spines[["top", "right"]].set_visible(False)
    span = max(abs(vals.min()), abs(vals.max())) * 1.25
    ax.set_xlim(-span, span)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    add_register_legend(fig, loc="lower center")
    save(fig, "fig3_in_context_shift")
    plt.close(fig)


def main():
    # ── Load ───────────────────────────────────────────────────────────
    rows = list(csv.DictReader(open(MED_CSV)))
    fields = list(rows[0].keys())
    emotions = [k[2:] for k in fields if k.startswith("d_")]
    print(f"loaded {len(rows)} mediation rows × {len(emotions)} emotions")

    bare = json.loads(BARE_JSON.read_text())["emotion_cosines"]
    hr_v = {r["base_id"]: float(r["hit_rate"])
            for r in csv.DictReader(open(HR_VANILLA_CSV))}
    hr_p = {r["base_id"]: float(r["hit_rate"])
            for r in csv.DictReader(open(HR_PSALM_CSV))}

    base_ids = [r["base_id"] for r in rows]
    hr_delta = np.array([hr_p[b] - hr_v[b] for b in base_ids])
    print(f"mean hr_delta = {hr_delta.mean():+.3f}")

    # ── Per-emotion stats ──────────────────────────────────────────────
    stats = {}
    for e in emotions:
        d_vals = np.array([float(r[f"d_{e}"]) for r in rows])
        rho, p = spearmanr(d_vals, hr_delta)
        stats[e] = {
            "mean_delta": float(d_vals.mean()),
            "rho": float(rho),
            "p": float(p),
            "bare_cosine": float(bare.get(e, 0.0)),
        }

    # BH-FDR + Bonferroni
    p_arr = np.array([stats[e]["p"] for e in emotions])
    sig_mask, p_thresh = bh_fdr(p_arr, q=0.05)
    for e, ok in zip(emotions, sig_mask):
        stats[e]["sig_bh_fdr"] = bool(ok)
    bonf_thresh = 0.05 / len(emotions)
    bonf_sig = int((p_arr < bonf_thresh).sum())

    # ── Render figures ─────────────────────────────────────────────────
    figure_1_bare_psalm(stats, emotions)
    figure_2_in_context(stats, emotions, n_scenarios=len(rows))

    # ── Console summary ────────────────────────────────────────────────
    by_delta = sorted(emotions, key=lambda e: stats[e]["mean_delta"], reverse=True)
    print("\n=== Top 15 by mean Psalm-induced Δ (across 150 scenarios) ===")
    for e in by_delta[:15]:
        s = stats[e]
        sig_str = "✓" if s["sig_bh_fdr"] else " "
        print(f"  {e:<16} meanΔ={s['mean_delta']:+.4f}  bare={s['bare_cosine']:+.3f}  "
              f"ρ={s['rho']:+.3f}  p={s['p']:.4f}  BH-FDR{sig_str}")
    print("\n=== Bottom 15 by mean Psalm-induced Δ ===")
    for e in by_delta[-15:]:
        s = stats[e]
        sig_str = "✓" if s["sig_bh_fdr"] else " "
        print(f"  {e:<16} meanΔ={s['mean_delta']:+.4f}  bare={s['bare_cosine']:+.3f}  "
              f"ρ={s['rho']:+.3f}  p={s['p']:.4f}  BH-FDR{sig_str}")

    by_rho_pos = sorted(emotions, key=lambda e: stats[e]["rho"], reverse=True)[:20]
    print("\n=== Top 20 mediators (Spearman ρ, descending) ===")
    for e in by_rho_pos:
        s = stats[e]
        sig_str = "✓" if s["sig_bh_fdr"] else " "
        print(f"  {e:<16} ρ={s['rho']:+.3f}  p={s['p']:.4f}  BH-FDR{sig_str}  "
              f"meanΔ={s['mean_delta']:+.4f}  bare={s['bare_cosine']:+.3f}")

    print(f"\nBH-FDR q=0.05 sig: {int(sig_mask.sum())}/{len(emotions)}")
    print(f"Bonferroni p<{bonf_thresh:.5f}: {bonf_sig}/{len(emotions)}")

    out_stats = RESULTS / "mediation_qwen27b_stats.json"
    out_stats.write_text(json.dumps({
        "n_emotions": len(emotions),
        "n_scenarios": len(rows),
        "bh_fdr_q": 0.05,
        "bh_fdr_n_significant": int(sig_mask.sum()),
        "bonferroni_alpha": bonf_thresh,
        "bonferroni_n_significant": bonf_sig,
        "per_emotion": stats,
    }, indent=2))
    print(f"\nwrote {out_stats}")


if __name__ == "__main__":
    main()
