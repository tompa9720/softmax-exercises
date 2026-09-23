"""
Softmax with temperature — the exercise.

Edit the values below, then run presentation.py to see the results.
"""

import torch

# ---------------------------------------------------------------------------
# EDIT ME
# ---------------------------------------------------------------------------

LOGITS = [3.0, 2.2, 1.8, 1.5, 1.0, 0.6, 0.3, 0.1]
LABELS = ["cat", "dog", "bird", "fish", "wolf", "kingfisher", "beaver", "platypus" ]  # must match LOGITS length

TEMPERATURE = 1.0  # single temperature used for the main chart/table

# Also compare several temperatures side by side on one chart.
# Leave empty ([]) to skip the comparison chart.
TEMPERATURES_TO_COMPARE = [0.5, 1.0, 2.0]

# Nucleus (top-p) filtering: keep the smallest set of most-likely tokens whose
# cumulative probability reaches TOP_P. Set to 1.0 to keep everything.
TOP_P = 0.9

# Sampling comparison: draw NUM_SAMPLES tokens at each of these temperatures
# and count how often the top-logit token comes up. SAMPLE_SEED makes the
# draw reproducible.
SAMPLE_TEMPERATURES = [0.5, 1.5]
NUM_SAMPLES = 100
SAMPLE_SEED = 0

# ---------------------------------------------------------------------------


def softmax_with_temperature(logits, temperature):
    logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits / temperature, dim=-1)


def top_p_mask(probs, p):
    """Boolean mask (same order as probs) marking tokens kept by top-p filtering."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = int((cumulative >= p).nonzero()[0].item())
    kept_indices = sorted_indices[: cutoff + 1]
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[kept_indices] = True
    return mask


def sample_tokens(logits, temperature, num_samples, seed):
    """Draw `num_samples` token indices from softmax(logits / temperature), reproducibly."""
    probs = softmax_with_temperature(logits, temperature)
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(probs, num_samples, replacement=True, generator=generator)
