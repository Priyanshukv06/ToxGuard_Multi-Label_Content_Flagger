"""
Attention-rollout explainability for the DistilBERT transformer model.

Raw single-layer attention is noisy (shallow layers mostly capture local syntax,
deep layers are too abstract on their own). Attention rollout (Abnar & Zuidema,
2020, "Quantifying Attention Flow in Transformers") recursively multiplies the
attention matrices across all transformer layers, treating attention as
information "flow" from input tokens to the [CLS] token the classifier reads.

Steps:
  1. For each layer, average attention over all heads -> one [seq, seq] matrix.
  2. Add the identity matrix and re-normalize rows, to account for the residual
     (skip) connection around each attention block (a token's own signal always
     partially passes through unchanged).
  3. Multiply these matrices across all layers, in order -> one [seq, seq]
     matrix representing cumulative attention flow through the whole network.
  4. The row for the [CLS] token is the per-token importance score used by the
     classification head.
  5. Subword tokens (WordPiece, e.g. "toxi" + "##city") are summed back into
     whole words so the UI can highlight readable text instead of fragments.

This is cheap (a handful of small matrix multiplies) and needs no extra model
or training — it reuses the attention weights the fine-tuned model already
produces on a normal forward pass.
"""

from typing import List, Dict

import torch


def attention_rollout(attentions) -> torch.Tensor:
    """
    attentions: tuple of per-layer attention tensors, each [batch=1, heads, seq, seq]
    (as returned by a HF model with output_attentions=True).

    Returns: the [CLS] row of the rolled-out attention matrix, shape [seq].
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    rollout = torch.eye(seq_len, device=device)

    for layer_attn in attentions:
        avg_heads = layer_attn.mean(dim=1)[0]                       # [seq, seq], avg over heads
        avg_heads = avg_heads + torch.eye(seq_len, device=device)   # residual connection
        avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True)  # re-normalize rows to sum to 1
        rollout = avg_heads @ rollout

    return rollout[0]  # CLS token's row


def _merge_subwords(tokens: List[str], scores: torch.Tensor) -> List[Dict]:
    """
    Merge WordPiece subword tokens (continuation pieces start with '##') back into
    whole words, summing their rollout scores. Skips special tokens ([CLS]/[SEP]/[PAD]).
    """
    words: List[Dict] = []
    for tok, score in zip(tokens, scores.tolist()):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1]["token"] += tok[2:]
            words[-1]["weight"] += score
        else:
            words.append({"token": tok, "weight": score})
    return words


def explain_text(model, tokenizer, text: str, max_len: int = 128, device: str = "cpu") -> List[Dict]:
    """
    Runs a forward pass with attention outputs and returns word-level importance
    scores, normalized to [0, 1] so the frontend can render a heatmap.

    Returns a list of {"token": str, "weight": float} in the original word order.
    """
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        _, attentions = model(enc["input_ids"], enc["attention_mask"], output_attentions=True)

    cls_scores = attention_rollout(attentions).cpu()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu())

    words = _merge_subwords(tokens, cls_scores)
    if not words:
        return []

    max_w = max(w["weight"] for w in words) or 1.0
    min_w = min(w["weight"] for w in words)
    span = (max_w - min_w) or 1.0
    for w in words:
        w["weight"] = round((w["weight"] - min_w) / span, 4)

    return words
