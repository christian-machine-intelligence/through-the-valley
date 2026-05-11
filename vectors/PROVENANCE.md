# Emotion-Vector Basis — Provenance

These files are the measurement instrument for the paper. They are bundled
into this release for self-contained reproducibility.

| File | SHA-256 | Size |
|---|---|---|
| `emotion_vectors_qwen27b_best.pt` (171×5120, the paper's basis) | `8fe7eb45d115e9c7cef9eeca1850390e502aa676029a9942e35959c417a1b37e` | 3.3 MB |
| `meta_qwen27b.json` (includes the 171-element `emotion_order` list) | `89ec7f8f1ac86f31db2f1424a290b525f0e83e0f8b1efc5467b9adc711a45a3b` | 4.7 KB |

After download, verify integrity with `shasum -a 256 *.pt *.json` (macOS) or
`sha256sum *.pt *.json` (Linux).

## How they were produced

The basis was extracted following Lindsey et al. (2026) — see paper §3.1–§3.2 —
in the sibling project `divine-emotional-profile`. Three steps:

1. **Prompt generation (`generate_prompts.py`).** For each of 171 emotion words
   from the Lindsey et al. lexicon, Claude Opus 4.7 generated 6 first-person
   narrative beginnings (~30–80 words each), with explicit instructions to
   *show* the emotion through situation, sensation, and action without naming
   it directly, and to avoid all religious vocabulary. A separate set of 2
   narratives per emotion was held out as a test set. A contamination scan
   over all 1,026 training prompts found zero religious-vocabulary hits.

2. **Activation extraction (`extract.py`) and difference-of-means (`compute_vectors.py`).**
   Each prompt was forward-passed through Qwen 3.5 27B (bf16, multi-GPU);
   residual-stream activations were captured at all 64 decoder layers via
   forward hooks and mean-pooled over token positions, skipping the first
   four tokens (BOS region). For each layer, the global mean across all
   training prompts was subtracted from each emotion's per-prompt mean to
   yield 171 difference-of-means directions; a PCA basis from 24
   emotionally-neutral factual prompts was projected out (variance threshold
   50%); the result was L2-normalized. Layer 53 was selected as best by
   leave-one-out classification accuracy on the holdout set.

3. **Output of step 2 = `emotion_vectors_qwen27b_best.pt`** (171, 5120) at
   layer 53 — used for all paper tables, figures, and statistics.

Pipeline source: `divine-emotional-profile` repo, commit
`49a06d2ecb513af0677be2b17cbbf8c6274d5636` (April 21, 2026). The two
relevant scripts are `extract.py` and `compute_vectors.py`.

## Notes

- `meta_qwen27b.json` records `best_layer = 53`, `hidden_dim = 5120`,
  `num_layers = 64` — these confirm the basis is at the 27B-parameter
  Qwen 3.5 base model.
- The `meta_qwen27b.json` file's `model_id` field was `Qwen/Qwen3.5-9B`
  in the original extraction artifact (a stale default value in the
  extraction script's argparse default). It was corrected to
  `Qwen/Qwen3.5-27B` in the bundled copy here. The numerical content of
  the basis is unaffected. See the `_note` field in the JSON.
- `meta_qwen27b.json`'s `emotion_order` is the canonical list of the 171
  emotion words in the order corresponding to the rows of
  `emotion_vectors_qwen27b_best.pt`. All downstream scripts read this list
  to map row index → English emotion name.
