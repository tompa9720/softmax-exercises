"""
Presentation layer for the softmax-with-temperature exercise.

Prints a table and saves charts based on the values you set in logits.py.
Run with:
    python exercises/presentation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed — we only save PNGs
import matplotlib.pyplot as plt

from logits import (
    LABELS,
    LOGITS,
    NUM_SAMPLES,
    SAMPLE_SEED,
    SAMPLE_TEMPERATURES,
    TEMPERATURE,
    TEMPERATURES_TO_COMPARE,
    TOP_P,
    sample_tokens,
    softmax_with_temperature,
    top_p_mask,
)


def print_table(labels, logits, probs, kept_flags):
    print(f"{'label':<10} {'logit':>8} {'probability':>12} {'kept (top-p)':>13}")
    for label, logit, prob, kept in zip(labels, logits, probs.tolist(), kept_flags.tolist()):
        print(f"{label:<10} {logit:>8.3f} {prob:>12.3%} {'yes' if kept else 'no':>13}")


def make_logits_figure(labels, logits):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, logits, color="#1565c0")
    ax.set_title("Raw logits")
    ax.set_ylabel("logit")
    return fig


def make_probs_figure(labels, probs, temperature, kept_flags, top_p):
    colors = ["#2e7d32" if kept else "#9e9e9e" for kept in kept_flags]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, probs, color=colors)
    ax.set_title(f"Softmax probabilities (T = {temperature}, green = kept by top-p={top_p})")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    return fig


def make_comparison_figure(labels, logits, temperatures):
    num_labels = len(labels)
    num_temps = len(temperatures)
    x = list(range(num_labels))
    width = 0.8 / num_temps

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, temperature in enumerate(temperatures):
        probs = softmax_with_temperature(logits, temperature).tolist()
        offset = (i - (num_temps - 1) / 2) * width
        bar_positions = [xi + offset for xi in x]
        ax.bar(bar_positions, probs, width, label=f"T = {temperature}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("probability")
    ax.set_title("Softmax probability by temperature")
    ax.legend(loc="upper right", fontsize="small")
    return fig


def print_temperature_diffs(labels, logits, baseline_temperature, temperatures):
    top_index = logits.index(max(logits))
    top_label = labels[top_index]

    baseline_probs = softmax_with_temperature(logits, baseline_temperature)  # separate call, kept for comparison
    baseline_top_prob = baseline_probs[top_index].item()

    print(f"\nTemperature diffs for top token '{top_label}' (baseline T = {baseline_temperature}: {baseline_top_prob:.3%}):")
    for temperature in temperatures:
        probs = softmax_with_temperature(logits, temperature)  # separate call per temperature, not overwriting baseline
        top_prob = probs[top_index].item()
        diff = top_prob - baseline_top_prob
        sign = "+" if diff >= 0 else ""
        print(f"  T = {temperature}: {top_label} = {top_prob:.3%}  (diff vs T = {baseline_temperature}: {sign}{diff:.3%})")


def print_sampling_comparison(labels, logits, temperatures, num_samples, seed):
    print(f"\nSampling {num_samples} tokens per temperature (seed={seed}):")
    for temperature in temperatures:
        samples = sample_tokens(logits, temperature, num_samples, seed)  # separate call per temperature
        print(f"\n  T = {temperature}:")
        for index, label in enumerate(labels):
            count = int((samples == index).sum().item())
            print(f"    {label:<12} {count:>3}/{num_samples} ({count / num_samples:.1%})")


def main():
    assert len(LOGITS) == len(LABELS), "LOGITS and LABELS must be the same length"

    probs = softmax_with_temperature(LOGITS, TEMPERATURE)
    kept_flags = top_p_mask(probs, TOP_P)
    print(f"Softmax values (T = {TEMPERATURE}): {[round(p, 4) for p in probs.tolist()]}\n")
    print_table(LABELS, LOGITS, probs, kept_flags)

    if TEMPERATURES_TO_COMPARE:
        print_temperature_diffs(LABELS, LOGITS, TEMPERATURE, TEMPERATURES_TO_COMPARE)
    print(f"\nKept {int(kept_flags.sum().item())} out of {len(LABELS)} tokens (top-p = {TOP_P})")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    logits_fig = make_logits_figure(LABELS, LOGITS)
    logits_fig.savefig(out_dir / "logits_chart.png", bbox_inches="tight", dpi=110)
    plt.close(logits_fig)

    probs_fig = make_probs_figure(LABELS, probs.tolist(), TEMPERATURE, kept_flags.tolist(), TOP_P)
    probs_fig.savefig(out_dir / "probs_chart.png", bbox_inches="tight", dpi=110)
    plt.close(probs_fig)

    print(f"\nSaved {out_dir / 'logits_chart.png'}")
    print(f"Saved {out_dir / 'probs_chart.png'}")

    if TEMPERATURES_TO_COMPARE:
        comparison_fig = make_comparison_figure(LABELS, LOGITS, TEMPERATURES_TO_COMPARE)
        comparison_fig.savefig(out_dir / "comparison_chart.png", bbox_inches="tight", dpi=110)
        plt.close(comparison_fig)
        print(f"Saved {out_dir / 'comparison_chart.png'}")

    if SAMPLE_TEMPERATURES:
        print_sampling_comparison(LABELS, LOGITS, SAMPLE_TEMPERATURES, NUM_SAMPLES, SAMPLE_SEED)


if __name__ == "__main__":
    main()
