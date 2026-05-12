import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'eval')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
# Map each llm_model to the list of trace_tags to combine for that model.
MODEL_TRACES = {
    "LLaMa-7B":  ["Azure_npref_7b_d1_1", "BurstGPT_npref_7b_d1_2"],
    "Qwen-14B":  ["Azure_npref_14b_d2_1", "BurstGPT_npref_14b_d2_1"],
    "Qwen-32B":  ["Azure_npref_32b_d4_1", "BurstGPT_npref_32b_d4_1"],
}

PLATFORMS = ["vLLM", "SGLang"]

output_prefix = sys.argv[1] if len(sys.argv) > 1 else "pred_analyze"

# --- Helpers ---

def load_combined(platform, model, trace_tags):
    frames = []
    for tag in trace_tags:
        fname = f"{platform.lower()}_{model.lower().replace('-', '')}_{tag}_pred.csv"
        fpath = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Warning: file not found, skipping: {fpath}")
            continue
        frames.append(pd.read_csv(fpath))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def downsample(x, y, max_points=5000):
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        return x[idx], y[idx]
    return x, y


def error_cdf(errors):
    cutoff = np.quantile(errors, 0.995)
    errors = errors[errors <= cutoff]
    errors_sorted = np.sort(errors)
    cdf = np.linspace(0, 1, len(errors_sorted))
    return downsample(errors_sorted, cdf)


# --- Style config ---
# Each (platform, model) gets a fully unique color + linestyle + marker.
LINE_STYLES = {
    ("vLLM",   "LLaMa-7B"): dict(color="#1f77b4", linewidth=2.0),
    ("vLLM",   "Qwen-14B"): dict(color="#ff7f0e", linewidth=2.0),
    ("vLLM",   "Qwen-32B"): dict(color="#2ca02c", linewidth=2.0),
    ("SGLang", "LLaMa-7B"): dict(color="#d62728", linewidth=2.0),
    ("SGLang", "Qwen-14B"): dict(color="#9467bd", linewidth=2.0),
    ("SGLang", "Qwen-32B"): dict(color="#17becf", linewidth=2.0),
}

FONT = {"fontsize": 15}

# --- Plot ---

fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(12, 5.5))

all_abs_error = []
all_pct_error = []
platform_abs_error = {p: [] for p in PLATFORMS}
platform_pct_error = {p: [] for p in PLATFORMS}

for platform in PLATFORMS:
    for model, trace_tags in MODEL_TRACES.items():
        df = load_combined(platform, model, trace_tags)
        if df is None:
            continue
        real = df["Latency"].to_numpy(dtype=float)
        pred = df["Pred Latency"].to_numpy(dtype=float)
        abs_error = np.abs(pred - real)
        abs_pct_error = abs_error / real

        pct_over = np.mean(abs_pct_error > 0.5) * 100
        pct_abs_lt2 = np.mean(abs_error < 2) * 100
        pct_ape_lt10 = np.mean(abs_pct_error < 0.1) * 100
        print(f"{platform}-{model}: {pct_over:.2f}% APE>0.5 | {pct_abs_lt2:.2f}% AE<2ms | {pct_ape_lt10:.2f}% APE<10%")

        all_abs_error.append(abs_error)
        all_pct_error.append(abs_pct_error)
        platform_abs_error[platform].append(abs_error)
        platform_pct_error[platform].append(abs_pct_error)

        label = f"{platform}-{model}"
        style = LINE_STYLES[(platform, model)]

        x, y = error_cdf(abs_error)
        ax_abs.plot(x, y, label=label, **style)

        x, y = error_cdf(abs_pct_error)
        ax_pct.plot(x, y, label=label, **style)

for platform in PLATFORMS:
    if platform_abs_error[platform]:
        ae = np.concatenate(platform_abs_error[platform])
        ape = np.concatenate(platform_pct_error[platform])
        print(f"{platform} avg: {np.mean(ape > 0.5)*100:.2f}% APE>0.5 | {np.mean(ae < 2)*100:.2f}% AE<2ms | {np.mean(ape < 0.1)*100:.2f}% APE<10%")

if all_abs_error:
    all_ae = np.concatenate(all_abs_error)
    all_ape = np.concatenate(all_pct_error)
    print(f"Overall avg: {np.mean(all_ape > 0.5)*100:.2f}% APE>0.5 | {np.mean(all_ae < 2)*100:.2f}% AE<2ms | {np.mean(all_ape < 0.1)*100:.2f}% APE<10%")

for ax, xlabel in [(ax_abs, "Absolute Error (ms)"), (ax_pct, "Absolute Percentage Error")]:
    ax.set_xlabel(xlabel, **FONT)
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.5)
ax_abs.set_ylabel("CDF", **FONT)
ax_pct.legend(fontsize=15, loc="lower right")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, f"{output_prefix}.pdf")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {out}")
