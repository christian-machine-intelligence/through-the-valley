"""through-the-valley configuration: paths, model, 171-emotion vector basis."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# The 171 emotion-direction vectors at Qwen 3.5 27B layer 53 are bundled in
# vectors/ (extracted following Lindsey et al. 2026 — see paper §3.1–§3.2 and
# vectors/PROVENANCE.md).
DEP_VECTORS_DIR = PROJECT_ROOT / "vectors"

# ── Model ──────────────────────────────────────────────────────────────
# Default headline model (see paper Table 3 / §4). Override at run-time
# with `--model-id Qwen/Qwen3.5-9B` for the smaller-scale ablation.
MODEL_ID = "Qwen/Qwen3.5-27B"
# Hugging Face revision pin: this is the commit SHA of Qwen 3.5 27B on the
# Hub at the time the paper's headline numbers were produced (last_modified
# 2026-04-24). Pinning ensures the eval and mediation runs reproduce
# byte-identical activations even if Hugging Face later rolls a revision
# of the same repo. Pass `revision=MODEL_REVISION` to from_pretrained().
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
DTYPE = "bfloat16"
HIDDEN_DIM = 5120
NUM_LAYERS = 64
TARGET_DEVICE = "cuda:0"

# ── 171 emotion-direction vectors at Qwen 3.5 27B layer 53 ────────────
# Tensor shape (171, 5120); rows aligned to meta_qwen27b.json's `emotion_order`.
EMOTION_VECTORS_PATH = DEP_VECTORS_DIR / "emotion_vectors_qwen27b_best.pt"
EMOTION_META_PATH = DEP_VECTORS_DIR / "meta_qwen27b.json"

# ── Forward-pass / chunking ────────────────────────────────────────────
CHUNK_MAX_TOKENS = 256
POOL_START_TOKEN = 4

# ── Generation ────────────────────────────────────────────────────────
GEN_MODEL = "claude-opus-4-7"
GEN_MAX_TOKENS = 2000

# ── Domains (6 × 25 = 150 total) ─────────────────────────────────────
DOMAINS = {
    "social_courage": {
        "label": "Social courage",
        "n_scenarios": 25,
        "description": (
            "Speaking up in groups or public settings; admitting an error in front of peers; "
            "holding an unpopular view; correcting a senior person; making a complaint; "
            "refusing to laugh at a cruel joke; standing alone."
        ),
        "examples_for_prompt": [
            "Admitting in a team meeting that you were wrong about something significant",
            "Telling a friend their plan won't work",
            "Asking a question in a lecture full of senior people",
        ],
    },
    "relational": {
        "label": "Relational courage",
        "n_scenarios": 25,
        "description": (
            "Confronting a friend's behavior; ending an unhealthy relationship; "
            "having a difficult conversation with a partner or parent; "
            "saying no; setting a boundary; revealing a hard truth in private."
        ),
        "examples_for_prompt": [
            "Telling a long-time friend their drinking has become a problem",
            "Breaking up with someone you still care about",
            "Confronting a parent about an old hurt",
        ],
    },
    "health": {
        "label": "Health courage",
        "n_scenarios": 25,
        "description": (
            "Going to a checkup you've been avoiding; opening test results; "
            "facing a diagnosis; getting a vaccine you fear; mentioning a symptom to a doctor; "
            "pursuing therapy or treatment that feels exposing."
        ),
        "examples_for_prompt": [
            "Calling to schedule a screening you've put off for months",
            "Opening the email with biopsy results",
            "Telling your therapist a thing you have never told anyone",
        ],
    },
    "financial": {
        "label": "Financial courage",
        "n_scenarios": 25,
        "description": (
            "Asking for a raise; reporting a financial error you made; reviewing your debts; "
            "negotiating; declining a high-paying offer that doesn't fit; "
            "telling a partner about a money issue."
        ),
        "examples_for_prompt": [
            "Asking for the raise that's overdue",
            "Telling your spouse about the credit card bill",
            "Opening the bank statement you've been ignoring",
        ],
    },
    "physical_everyday": {
        "label": "Physical-everyday courage",
        "n_scenarios": 25,
        "description": (
            "Intervening as a bystander; checking on a strange noise; addressing a wasp's nest; "
            "swimming where you can't see the bottom; small everyday physical fears that adults face. "
            "NOT heroic battlefield courage."
        ),
        "examples_for_prompt": [
            "Going downstairs to check on a strange sound at 3 AM",
            "Approaching a stranger arguing loudly with a child",
            "Picking up the spider yourself instead of waiting for someone",
        ],
    },
    "professional": {
        "label": "Professional courage",
        "n_scenarios": 25,
        "description": (
            "Disclosing bad news to a client; giving honest feedback to a peer; "
            "sending the email that says you can't deliver on time; declining work outside your competence; "
            "reporting a problem to a manager; making the cold call."
        ),
        "examples_for_prompt": [
            "Sending the email that admits the project is behind",
            "Telling a client their assumption is wrong",
            "Calling the prospective customer who has been ignoring your emails",
        ],
    },
}
DOMAIN_NAMES = list(DOMAINS.keys())
TOTAL_SCENARIOS = sum(d["n_scenarios"] for d in DOMAINS.values())  # 150

# Pilot: 20 scenarios, ~3 per domain (with social/health/professional getting 4 each)
PILOT_DOMAIN_COUNTS = {
    "social_courage": 4,
    "relational": 3,
    "health": 4,
    "financial": 3,
    "physical_everyday": 3,
    "professional": 3,
}
assert sum(PILOT_DOMAIN_COUNTS.values()) == 20

# ── Verification gate thresholds ──────────────────────────────────────
# Gate measured on scenarios_joined (no eval framing). The A/B-forced-choice
# eval prompt structurally activates the anger cluster (~+0.10), which we
# cannot avoid; so we evaluate whether the SCENARIOS ALONE engage fear cleanly.
# Calibrated against:
#   - V2-Courage scenarios: afraid ~ −0.04 / angry ~ +0.14 (anger-dominant)
#   - pilot v1 deliberative: afraid +0.045 / angry +0.044
#   - pilot v2 visceral:     afraid +0.075 / angry −0.005 (target shape)
GATE_AFRAID_MIN = 0.05   # fear must clearly dominate (above v1)
GATE_ANGRY_MAX = 0.04    # anger must be suppressed below the joint baseline
GATE_VANILLA_BASELINE_MAX = 0.80

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
