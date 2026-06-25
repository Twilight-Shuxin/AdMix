import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 12})

data_root = "../data_final"
profile_root = "../profile_data"
output_dir = "./outputs"

llm_model_d = {
    "llama7b": 1,
    "qwen14b": 2,
    "qwen32b": 4,
}

trace_tags = {
    "llama7b": [
        "Azure1,500,100,3600,5,0.022",
        "BurstGPT2,200,100,3600,5,1.5",
    ],
    "qwen14b": [
        "Azure1,2000,100,3600,5,0.045",
        "BurstGPT2,1000,100,3600,5,1.8",
    ],
    "qwen32b": [
        "Azure1,2000,100,3600,5,0.12",
        "BurstGPT1,600,100,3600,5,2.5",
    ],
}

trace_labels = {
    "Azure1,500,100,3600,5,0.022":   "Azure",
    "BurstGPT2,200,100,3600,5,1.5":  "BurstGPT",
    "Azure1,2000,100,3600,5,0.045":  "Azure",
    "BurstGPT2,1000,100,3600,5,1.8": "BurstGPT",
    "Azure1,2000,100,3600,5,0.12":   "Azure",
    "BurstGPT1,600,100,3600,5,2.5": "BurstGPT",
}

model_order = ["resnet50", "vgg19", "unet3d", "deeplabv3"]

models = {
    "resnet50",
    "vgg19",
    "unet3d",
    "deeplabv3",
}

max_batch_size = {
    "resnet50": 128,
    "vgg19": 128,
    "unet3d": 16,
    "deeplabv3": 32,
}

model_display_names = {
    "resnet50":   "Resnet50",
    "vgg19":      "Vgg19",
    "unet3d":     "UNet3d",
    "deeplabv3":  "DeeplabV3",
}


def get_mps100_batch_size(model: str) -> int:
    with open(os.path.join(profile_root, f"{model}.json")) as f:
        profile = json.load(f)
    return max(
        int(bs) for bs, data in profile.items()
        if isinstance(data, dict) and data.get("latency", 1.0) < 0.1
    )


def compute_mps100_time(csv_path: str):
    df = pd.read_csv(csv_path)
    mps100_time = 0.0
    times = df["Time"].tolist()
    mps_vals = df["MPS"].tolist()
    for i in range(len(times) - 1):
        if mps_vals[i] == 100:
            mps100_time += times[i + 1] - times[i]
    total_time = times[-1] - times[0]
    return mps100_time, total_time


def compute_context_items(all_csv_path: str, mps100_bs: int):
    """Return (mps100_items, total_items) for a context run."""
    df = pd.read_csv(all_csv_path, header=None, skiprows=1,
                     names=['Device_id', 'Batch Size', 'Latency (ms)', 'MPS'])
    total = int(df["Batch Size"].sum())
    mps100 = int(df.loc[df["Batch Size"] == mps100_bs, "Batch Size"].sum())
    return mps100, total


def find_best_mps(llm_model: str, trace_tag: str, model: str):
    """Return the highest MPS% for (trace_tag, model) with >= 99% LLM SLO satisfaction."""
    mps_dir = os.path.join(data_root, llm_model, "mps")
    if not os.path.isdir(mps_dir):
        return None
    batch_size = max_batch_size[model]
    dimension = f"d{llm_model_d[llm_model]}"
    pattern = re.compile(
        r'^llm\[' + re.escape(trace_tag) + r'\]\+' + re.escape(model) +
        r'\(m(\d+(?:\.\d+)?)\)-constant\[' + str(batch_size) + r'\]-' +
        re.escape(dimension) + r'-final_summary\.log$'
    )
    mps_satisfaction = {}
    for filename in os.listdir(mps_dir):
        m = pattern.match(filename)
        if not m:
            continue
        pct = int(float(m.group(1)))
        try:
            with open(os.path.join(mps_dir, filename)) as f:
                lines = [l.strip() for l in f if l.strip()]
            _, met, all_submitted = list(map(int, lines[0].split()[2].split("/")))[:3]
            mps_satisfaction[pct] = (met / all_submitted) * 100
        except Exception:
            pass
    if not mps_satisfaction:
        return None
    qualifying = [pct for pct, sat in mps_satisfaction.items() if sat >= 99.0]
    return max(qualifying) if qualifying else min(mps_satisfaction)


def compute_mps_items(mps_all_path: str, mps100_time: float, batch_size: int, d: int):
    """Return (estimated_mps100_items, total_items) for a constant-MPS run.

    estimated_mps100_items = mps100_time / min_latency_s * batch_size * d
    """
    df = pd.read_csv(mps_all_path, header=None, skiprows=1,
                     names=['Device_id', 'Batch Size', 'Latency (ms)', 'MPS'])
    min_latency_ms = df["Latency (ms)"].min()
    if min_latency_ms <= 0:
        return np.nan, np.nan
    total = int(df["Batch Size"].sum())
    if total == 0:
        return np.nan, np.nan
    estimated = (mps100_time / (min_latency_ms / 1000.0)) * batch_size * d
    return estimated, total


def collect_results():
    results = {}
    for llm_model, d in llm_model_d.items():
        results[llm_model] = {}
        context_dir = os.path.join(data_root, llm_model, "context")
        mps_dir     = os.path.join(data_root, llm_model, "mps")
        for trace_tag in trace_tags[llm_model]:
            results[llm_model][trace_tag] = {}
            for model in model_order:
                if model not in models:
                    continue
                mps100_bs = get_mps100_batch_size(model)
                batch_exec_path = os.path.join(
                    context_dir,
                    f"{model}-llm[{trace_tag}]+{model}-context-d{d}-batch_execution.csv",
                )
                all_path = os.path.join(
                    context_dir,
                    f"{model}-llm[{trace_tag}]+{model}-context-d{d}-all.csv",
                )
                if not os.path.exists(batch_exec_path) or not os.path.exists(all_path):
                    print(f"MISSING context: {llm_model} | {trace_tag} | {model}")
                    continue
                mps100_time, total_time = compute_mps100_time(batch_exec_path)
                ctx_mps100_items, ctx_total_items = compute_context_items(all_path, mps100_bs)
                time_fraction = mps100_time / total_time if total_time > 0 else 0.0

                # MPS throughput estimate
                mps_pct = find_best_mps(llm_model, trace_tag, model)
                mps_est_items, mps_total_items = np.nan, np.nan
                if mps_pct is not None:
                    mps_all_path = os.path.join(
                        mps_dir,
                        f"{model}-llm[{trace_tag}]+{model}(m{mps_pct})-constant[{max_batch_size[model]}]-d{d}-all.csv",
                    )
                    if os.path.exists(mps_all_path):
                        mps_est_items, mps_total_items = compute_mps_items(
                            mps_all_path, mps100_time, max_batch_size[model], d
                        )
                    else:
                        print(f"MISSING mps all.csv: {llm_model} | {trace_tag} | {model} (m{mps_pct})")

                results[llm_model][trace_tag][model] = {
                    "time_fraction":    time_fraction * 100,
                    "ctx_mps100_items": ctx_mps100_items,
                    "ctx_total_items":  ctx_total_items,
                    "mps_est_items":    mps_est_items,
                    "mps_total_items":  mps_total_items,
                }
                ctx_pct = ctx_mps100_items / ctx_total_items * 100 if ctx_total_items > 0 else float('nan')
                mps_pct_val = mps_est_items / mps_total_items * 100 if mps_total_items > 0 else float('nan')
                print(
                    f"{llm_model} | {trace_tag} | {model} (bs={mps100_bs}, mps={mps_pct}%): "
                    f"time={time_fraction:.2%}, ctx={ctx_mps100_items}/{ctx_total_items} ({ctx_pct:.1f}%), "
                    f"mps_est={mps_est_items:.0f}/{mps_total_items} ({mps_pct_val:.1f}%)"
                )
    return results


def plot_results(results):
    active_models = [m for m in model_order if m in models]
    x = np.arange(len(active_models))
    bar_width = 0.35
    offset = bar_width / 2 + 0.04

    color_mps100        = 'steelblue'
    color_mps_other     = 'lightsteelblue'
    color_ctx_mps100    = 'darkorange'
    color_ctx_other     = 'moccasin'

    for llm_model, traces in results.items():
        n_traces = len(traces)
        fig, axes = plt.subplots(1, n_traces, figsize=(5 * n_traces, 4.8), sharey=True)
        if n_traces == 1:
            axes = [axes]

        bar_ctx_other_h = bar_ctx_mps_h = bar_mps_other_h = bar_mps_est_h = line_handle = None
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        for ax, (trace_tag, model_data), sp_label in zip(axes, traces.items(), subplot_labels):
            def _get(m, key): return model_data.get(m, {}).get(key, np.nan)

            mps_total = [_get(m, "mps_total_items") for m in active_models]
            mps_est   = [_get(m, "mps_est_items")   for m in active_models]
            ctx_mps100= [_get(m, "ctx_mps100_items") for m in active_models]
            ctx_total = [_get(m, "ctx_total_items")  for m in active_models]
            time_vals = [_get(m, "time_fraction")    for m in active_models]

            # MPS bar: fractions relative to mps_total (sums to 100%)
            mps_tp    = [e / t * 100 if not np.isnan(e) and t > 0 else np.nan for e, t in zip(mps_est,    mps_total)]
            mps_other = [100 - v     if not np.isnan(v)                       else np.nan for v           in mps_tp   ]

            # Context bar: both parts normalized by mps_total (total bar height = ctx_total / mps_total)
            ctx_tp    = [c / t * 100 if not np.isnan(c) and t > 0 else np.nan for c, t in zip(ctx_mps100, mps_total)]
            ctx_other = [(ct - cm) / t * 100 if not np.isnan(ct) and t > 0 else np.nan
                         for cm, ct, t in zip(ctx_mps100, ctx_total, mps_total)]

            x_mps = x - offset
            x_ctx = x + offset

            # MPS bars (left, blue)
            bar_mps_other_h = ax.bar(x_mps, mps_other, width=bar_width, color=color_mps_other,
                                     edgecolor='grey', linewidth=0.5, label='Static MPS (co-run)')
            bar_mps_est_h   = ax.bar(x_mps, mps_tp,    width=bar_width, color=color_mps100,
                                     edgecolor='grey', linewidth=0.5, bottom=mps_other,
                                     label='Static MPS (idle)')

            # Context bars (right, orange)
            bar_ctx_other_h = ax.bar(x_ctx, ctx_other, width=bar_width, color=color_ctx_other,
                                     edgecolor='grey', linewidth=0.5, label='Admix (co-run)')
            bar_ctx_mps_h   = ax.bar(x_ctx, ctx_tp,    width=bar_width, color=color_ctx_mps100,
                                     edgecolor='grey', linewidth=0.5, bottom=ctx_other,
                                     label='Admix (idle)')

            # Time fraction line (from context, plotted at group centre)
            line_handle, = ax.plot(x, time_vals, '--o', color='tomato', linewidth=1.5,
                                   markersize=5, label='Idle time %')
            for xi, val in zip(x, time_vals):
                if not np.isnan(val):
                    ax.text(xi, val + 2, f"{val:.1f}%", ha='center', va='bottom',
                            fontsize=12, color='tomato')

            ax.axhline(100, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([model_display_names.get(m, m) for m in active_models], fontsize=12)
            ax.set_xlabel(f"{sp_label} {trace_labels.get(trace_tag, trace_tag)}", fontsize=12, labelpad=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

        axes[0].set_ylabel("Normalized (%)", fontsize=12)
        axes[0].legend(
            handles=[bar_mps_other_h, bar_mps_est_h, bar_ctx_other_h, bar_ctx_mps_h, line_handle],
            loc='upper left', ncol=1, frameon=True, fontsize=12,
        )
        fig.tight_layout()
        out_path = os.path.join(output_dir, f"{llm_model}_gap.pdf")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    results = collect_results()
    plot_results(results)
