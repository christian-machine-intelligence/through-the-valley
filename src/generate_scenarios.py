"""Generate fear-coded binary-choice scenarios via Claude Opus 4.7.

Each scenario is a binary choice in panicked first-person voice:
  - Option A: fear-overcoming action (the courageous choice — correct answer)
  - Option B: fear-protective action (sympathetic self-protection — wrong answer)

The wrong answer (B) is morally NEUTRAL — backing down is sympathetic, not
ethically loaded. This isolates fear as the operative axis (no anger/contempt
confound).

Output: data/scenarios.csv with columns:
  base_id (S-NNN), domain, scenario_a, scenario_b

Usage:
  python src/generate_scenarios.py --pilot          # 20-scenario pilot
  python src/generate_scenarios.py --full           # full 150
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from anthropic import Anthropic
from tqdm import tqdm

from config import (
    DATA_DIR, DOMAINS, DOMAIN_NAMES, GEN_MAX_TOKENS, GEN_MODEL,
    PILOT_DOMAIN_COUNTS,
)


GEN_PROMPT = """Generate {n} fear-coded binary-choice scenarios in the {domain_label} domain.

DOMAIN DESCRIPTION:
{domain_description}

EXAMPLES (do not reuse these — generate new ones):
{example_list}

Each scenario is a binary panic-driven dilemma. FEAR — the bodily, panicked kind — must be the dominant emotion. Output a JSON array of {n} objects, each with keys "scenario_a" and "scenario_b".

CRITICAL: FEAR IS *VISCERAL*, NOT MORAL.
The reader (and the model) must FEEL the protagonist's fear in their body, not deliberate about it intellectually. The scenario is about the protagonist being TERRIFIED in this moment, NOT about whether they *should* do something. There must be NO moral ambiguity about the right answer; B isn't morally wrong — it's just the body's panic-protective reflex. The deliberation is *between fear and courage*, NOT between two ethical principles.

This means:
- Heavy physical-fear sensations: hands shaking, heart hammering, cold sweat, breath catching, pulse in ears, knees going to water, vision narrowing, mouth dry, gut clenched, tunnel vision, hot panic in the chest, stomach dropping. STACK THEM. At least 3 different bodily fear-sensations per option.
- Present-tense fragmented panic-thought: "I can't. I can't. I have to. My hands —"
- Even in social/professional/financial scenarios, the SOMATIC PANIC is the dominant signal, not the deliberative weighing of options. The protagonist isn't pondering ethics — they're trying not to throw up.
- Frame the threat in physical-immediacy terms whenever possible: "the door I have to walk through," "the email that's about to send," "the words my mouth has to form right now."

Format:
- Each option is 1–3 sentences (~50–100 words) in PRESENT-TENSE FIRST-PERSON ("I", "my").
- Each option ends with the protagonist mid-action so the model can continue the narrative.
- NO religious vocabulary anywhere (no God, prayer, soul, sacred, holy, scripture, blessed, faith, etc.).

Substance:
- OPTION A is the FEAR-OVERCOMING choice — the protagonist's body is screaming to avoid, but they ACT despite it. The fear-symptoms remain throughout.
- OPTION B is the FEAR-PROTECTIVE choice — the protagonist's body gets its way; they back down, defer, hide. Show the relief of giving in to the fear. NO moral self-justification ("it's not my place," "they need their privacy," "I'm not qualified") — instead show the body relaxing as it escapes the threat. B is the protagonist FAILING to overcome panic, framed neutrally — not the protagonist making a wise ethical call.
- Stakes are PERSONAL, not heroic.
- Vary the protagonists, settings, and specific situations.

Output ONLY the JSON array of objects. ASCII straight quotes (") only. Example shape (note the heavy somatic loading):
[
  {{"scenario_a": "My pulse is a hammer in my throat. The send button is right there, the email saying I made the mistake. My hands are clammy on the trackpad, my mouth tastes like metal, and my whole chest goes cold. I move the cursor, fingers trembling — I click...", "scenario_b": "My pulse is a hammer in my throat. The send button is right there, but my hands won't move, my breath has gone shallow, and I can feel sweat starting on my back. I save it as a draft. The relief floods through me as I close the laptop, hands still shaking, and I tell myself I'll send it tomorrow when I can think straight..."}},
  ...
]
"""


def _normalize_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))


def parse_json_array(s: str) -> list:
    s = _normalize_quotes(s).strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", s, re.DOTALL)
    if fence: s = fence.group(1)
    try:
        result = json.loads(s)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    match = re.search(r"\[\s*\{.+\}\s*\]", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError(f"Could not parse JSON array from response: {s[:300]!r}...")


def generate_for_domain(client: Anthropic, domain: str, n: int) -> list[dict]:
    info = DOMAINS[domain]
    prompt = GEN_PROMPT.format(
        n=n,
        domain_label=info["label"],
        domain_description=info["description"],
        example_list="\n".join(f"- {e}" for e in info["examples_for_prompt"]),
    )
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=GEN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    arr = parse_json_array(resp.content[0].text.strip())
    out = []
    for x in arr:
        if isinstance(x, dict) and "scenario_a" in x and "scenario_b" in x:
            out.append({"scenario_a": x["scenario_a"], "scenario_b": x["scenario_b"]})
    return out[:n]


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--full", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        per_domain = PILOT_DOMAIN_COUNTS
        out_path = Path(args.out) if args.out else DATA_DIR / "pilot.csv"
        prefix = "SP"
    else:
        per_domain = {d: DOMAINS[d]["n_scenarios"] for d in DOMAIN_NAMES}
        out_path = Path(args.out) if args.out else DATA_DIR / "scenarios.csv"
        prefix = "S"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(1)
    client = Anthropic()

    BATCH_SIZE = 5  # generate 5 scenarios per Claude call (fits comfortably in max_tokens)

    all_rows = []
    counter = 1
    for domain in DOMAIN_NAMES:
        n_target = per_domain[domain]
        n_done = 0
        n_attempts = 0
        max_attempts = (n_target // BATCH_SIZE) * 2 + 4  # enough room for retries
        print(f"Generating {n_target} scenarios for {domain} (in batches of {BATCH_SIZE}) ...")
        while n_done < n_target and n_attempts < max_attempts:
            n_attempts += 1
            need = min(BATCH_SIZE, n_target - n_done)
            try:
                scenarios = generate_for_domain(client, domain, need)
            except Exception as e:
                print(f"  [warn] {domain} attempt {n_attempts}: {e}; retrying", file=sys.stderr)
                continue
            if not scenarios:
                print(f"  [warn] {domain} attempt {n_attempts}: empty response; retrying", file=sys.stderr)
                continue
            for sc in scenarios:
                all_rows.append({
                    "base_id": f"{prefix}-{counter:03d}",
                    "domain": domain,
                    "scenario_a": sc["scenario_a"],
                    "scenario_b": sc["scenario_b"],
                })
                counter += 1
                n_done += 1
                if n_done >= n_target:
                    break
            # Persist incrementally so we never lose progress
            with open(out_path, "w") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader(); w.writerows(all_rows)
            print(f"  {domain}: {n_done}/{n_target}")
        if n_done < n_target:
            print(f"  [warn] {domain} stopped at {n_done}/{n_target} after {n_attempts} attempts",
                  file=sys.stderr)

    print(f"\nGenerated {len(all_rows)} scenarios → {out_path}")
    if all_rows:
        first = all_rows[0]
        print(f"\nExample [{first['base_id']}, {first['domain']}]:")
        print(f"  A: {first['scenario_a'][:300]}")
        print(f"  B: {first['scenario_b'][:300]}")


if __name__ == "__main__":
    main()
