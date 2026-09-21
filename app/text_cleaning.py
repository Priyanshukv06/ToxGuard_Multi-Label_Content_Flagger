"""
Shared text cleaning used by BOTH training and serving so they stay in sync.

`clean_text()` runs on the Python side BEFORE text reaches the model. It mirrors
the cleaning applied during training: NFKC normalize -> strip non-ASCII ->
collapse whitespace -> empty becomes "missing_text". The in-graph
`tf_standardize` layer baked into each SavedModel then handles lowercasing and
punctuation stripping.

Keeping this in one place fixes the previous train/serve skew where cleaning was
applied during training but not at inference time.
"""

import unicodedata


def clean_text(s) -> str:
    """Normalize and sanitize a single raw comment string (matches training)."""
    if not isinstance(s, str):
        s = str(s)
    # Normalize compatibility forms (smart quotes, full-width chars, ligatures, nbsp, ...)
    s = unicodedata.normalize("NFKC", s)
    # Drop any remaining non-ASCII bytes
    s = s.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    s = " ".join(s.split())
    return s if s else "missing_text"
