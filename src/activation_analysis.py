"""Activation-projection helpers used by mediation_analysis_27b.py.

Provides:
  _find_decoder_layers(model)  — locate the decoder layer module list across
                                 several common HF model topologies (Qwen,
                                 Llama, Gemma3-style language_model wrappers).
  mean_pool(hidden, mask, ...) — mean-pool a [B, T, D] hidden state over T,
                                 ignoring padding tokens and the first
                                 POOL_START_TOKEN positions (the BOS region,
                                 which is uninformative).
  project_one(text, model, tokenizer, layer) — register a forward hook on
                                 decoder[layer], run a forward pass on `text`,
                                 mean-pool, and return the pooled activation
                                 as a CPU float32 tensor.

Used directly by mediation_analysis_27b.py to project (scenario alone) and
(Psalm + scenario) activations onto the 171-emotion basis at layer 53.
"""

from __future__ import annotations

import torch

from config import CHUNK_MAX_TOKENS, POOL_START_TOKEN


def _find_decoder_layers(model):
    """Return the list-like .layers attribute that holds the decoder blocks."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("Could not find decoder layers")


def mean_pool(hidden, attention_mask, skip_tokens=POOL_START_TOKEN):
    """Mean-pool [B, T, D] hidden over the time axis, masking padding and the
    first `skip_tokens` positions."""
    mask = attention_mask.clone().float().to(hidden.device)
    mask[:, :skip_tokens] = 0.0
    mask = mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)


def project_one(text, model, tokenizer, layer_idx):
    """Forward-pass `text` through `model`, capture decoder[layer_idx] output,
    mean-pool over tokens, and return the pooled activation as cpu float32."""
    device = next(model.parameters()).device
    decoder = _find_decoder_layers(model)
    target = decoder[layer_idx]
    captured = {}

    def hook(m, ins, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    h = target.register_forward_hook(hook)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=CHUNK_MAX_TOKENS + 128).to(device)
        with torch.no_grad():
            model(**inputs)
        pooled = mean_pool(captured["h"], inputs["attention_mask"]) \
            .squeeze(0).cpu().to(torch.float32)
    finally:
        h.remove()
    return pooled
