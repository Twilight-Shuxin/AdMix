import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D as MplLine2D
from matplotlib.container import BarContainer


class HandlerBarLine(HandlerBase):
    """Legend key that stacks the bar patch (top) over the line+marker (bottom)."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        bar_h, line_h = orig_handle

        # BarContainer — pull style from its first patch
        patch = bar_h.patches[0]

        # All coords must stay within [0, height].
        # Text baseline is at ydescent (≈5.95); below that is "below text."
        # Line goes below the text baseline; bar occupies the upper half.
        line_y     = height * 0.005           # ≈2.7, below text baseline (ydescent≈5.95)
        bar_bottom = height * 0.3           # ≈9.4, comfortably above the line
        bar_h_size = height * 0.38           # ≈6.9, bar_top≈16.3 (within height≈18)
        bar_y      = bar_bottom

        # Top — bar rectangle
        rect = Rectangle(
            (xdescent, bar_y),
            width, bar_h_size,
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            hatch=patch.get_hatch(),
            linewidth=patch.get_linewidth(),
            transform=trans,
            clip_on=True,
        )

        # Bottom — line with a single center marker (matches default HandlerLine2D behavior)
        mid_x = xdescent + width / 2
        line = MplLine2D(
            [xdescent, mid_x, xdescent + width], [line_y, line_y, line_y],
            color=line_h.get_color(),
            linestyle=line_h.get_linestyle(),
            linewidth=line_h.get_linewidth(),
            marker=line_h.get_marker(),
            markersize=max(line_h.get_markersize() * 0.8, 4),
            markevery=[1],
            transform=trans,
        )

        return [rect, line]

MODEL_DIMENSION = {
    "llama7b": "d1",
    "qwen14b": "d2",
    "qwen32b": "d4",
}
MODELS = ["llama7b", "qwen14b"]
# MODELS = ["qwen14b"]
ROOT_FOLDER = f"../data_final/{MODELS[0]}"
PROFILE_DATA_FOLDER = "../profile_data"
OUTPUT_FOLDER = "./outputs"
DIMENSION = MODEL_DIMENSION[MODELS[0]]

MPS_LOWER_LIMIT = 10  # ignore MPS percentage files below this value (0 = no limit)

config_map = {
    "time":    {"name": "Temporal Sharing", "scheduler": "constant"},
    "batch":   {"name": "batch",        "scheduler": "batch"},
    "mps":     {"name": "Static MPS",          "scheduler": "constant"},
    "pmps":     {"name": "Static MPS +",   "scheduler": "pconstant"},
    "contextminus": {"name": "AdMix -", "scheduler": "contextminus"},
    "utils":   {"name": "LLM only",     "scheduler": None},
    "context": {"name": "AdMix",      "scheduler": "context"},
}

max_batch_size = {
    "resnet50": 128,
    "vgg19": 128,
    "unet3d": 16,
    "deeplabv3": 32
}

time_batch_size = {
    "resnet50": 4,
    "vgg19": 4,
    "unet3d": 1,
    "deeplabv3": 2
}


transform_percentage_list = ['sm_active']

def load_profile_data(model):
    with open(f'{PROFILE_DATA_FOLDER}/{model}.json', 'r') as f:
        return json.load(f)

def read_final_summary(path, side_model = True):
    result = {}
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # ---- Parse the first line ----
    # Example: llm True 93774/94452/95265 ...
    first_parts = lines[0].split()
    counts = first_parts[2]  # "93774/94452/95265"
    _, met, all_submitted = list(map(int, counts.split("/")))[:3]

    result["met"] = met
    result["all_submitted"] = all_submitted
    result["satisfaction rate"] = round((met / all_submitted) * 100, 1)
    if(side_model):
        # ---- Parse second line ----
        # Example: unet3d 45232
        second_parts = lines[1].split()
        result["absolute throughput"] = int(second_parts[1])

    # ---- Parse sections separated by ==== ----
    sep_indices = [i for i, line in enumerate(lines) if line.startswith("====")]

    # First ==== section: DCGM key-value stats (optional)
    if len(sep_indices) >= 1:
        end = sep_indices[1] if len(sep_indices) >= 2 else len(lines)
        for line in lines[sep_indices[0] + 1:end]:
            key, value = line.split()
            value = float(value)
            if key in transform_percentage_list:
                value = round(value * 100, 1)
            result[key] = value

    # Second ==== section: energy line (optional), format: 29.67(kJ) 296(W) 100(s)
    if len(sep_indices) >= 2:
        for line in lines[sep_indices[1] + 1:]:
            parts = line.split()
            result["energy"] = float(parts[0].replace("(kJ)", ""))

    return result

"""
Read summary stats (avg, max, 99 percentile, etc)
"""
def read_stats(file_path, tag):
    result = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            value = float(parts[1])
            # replace % in keys (e.g., "90%") -> "p90"
            key = key.replace("%", "")
            if key.isdigit():
                key = f"p{key}"
            result[f"{tag}_{key}"] = value
    return result


def get_data(model, trace_tag, config, sub_path, percentage=None):
    data = {}
    scheduler = config_map[config]["scheduler"]
    if scheduler is not None:
        if scheduler in ("constant", "pconstant"):
            scheduler += f"[{max_batch_size[model]}]" if config in ("mps", "pmps") else f"[{time_batch_size[model]}]"
        set_tag = f'llm[{trace_tag}]+{model}'
        if config in ("mps", "pmps"):
            set_tag += f"(m{percentage})"
        set_tag += "-" + scheduler
        path_prefix = f"llm-{set_tag}"
        data.update(read_final_summary(f'{ROOT_FOLDER}/{sub_path}/{set_tag}-{DIMENSION}-final_summary.log'))
    else:
        path_prefix = "llm[" + trace_tag + "]"
        data.update(read_final_summary(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-final_summary.log', False))
    data.update(read_stats(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-ftsummary.log', "ft"))
    data.update(read_stats(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-btsummary.log', "bt"))
    return data

def plot_bar_chart(models, sub_paths, values, ylabel, ax=None, xtitle=None, ymax=None, ymin=None, show_legend=True):
    idx = None
    if(not ax):
        fig, ax = plt.subplots()
    else:
        plt.sca(ax)
        idx = ax.figure.axes.index(ax)

    if ymin or ymax:
        ax.set_ylim(
            ymin if ymin else ax.get_ylim()[0],
            ymax if ymax else ax.get_ylim()[1]
        )
    ymin_current, ymax_current = ax.get_ylim()
    y_range = ymax_current - ymin_current
    offset = 0.06 * y_range

    hatches = ['///', 'xx', '-', '\\\\', '+']
    x = np.arange(len(models))  # model positions
    bar_width = 0.8 / len(sub_paths)  # divide bar width by number of bars per group

    for i, config in enumerate(sub_paths):
        y = values[:, i]
        y_plot = np.clip(y,
            ymin if ymin is not None else -np.inf,
            ymax if ymax is not None else np.inf
        )
        ax.bar(x + i * bar_width, y_plot, hatch=hatches[i], edgecolor='white', width=bar_width, label=config_map[config]["name"])
        for xi, yi in zip(x, y):
            if ymax and (yi > ymax):
                ax.text(xi, ymax - offset, f"{yi:.1f}", ha='center', va='bottom', fontsize=8)
            if ymin and (yi < ymin):
                ax.text(xi, ymin + offset, f"{yi:.1f}",ha='center', va='top', fontsize=8)

    ax.set_xticks(x + bar_width * (len(sub_paths) / 2 - 0.5))
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    if(xtitle):
        ax.set_xlabel(xtitle)
    if show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), columnspacing=1, ncol=len(sub_paths), frameon=True)

def plot_line_chart(models, sub_paths, values, ylabel, ax=None, xtitle=None, ymax=None, ymin=None, show_legend=True):
    x = np.arange(len(models))  # model positions
    styles = ['--x', ':s', '-o', '-.^',  '-d', '-*']  # different line styles and markers for each config
    if(ax == None):
        fig, ax = plt.subplots()
    else:
        plt.sca(ax)
        idx = ax.figure.axes.index(ax)

    if ymin or ymax:
        ax.set_ylim(
            ymin if ymin else ax.get_ylim()[0],
            ymax if ymax else ax.get_ylim()[1]
        )
    ymin_current, ymax_current = ax.get_ylim()
    y_range = ymax_current - ymin_current
    offset = 0.06 * y_range

    for i, config in enumerate(sub_paths):
        y = values[:, i]
        y_plot = np.clip(y,
            ymin if ymin is not None else -np.inf,
            ymax if ymax is not None else np.inf
        )
        ax.plot(x, y_plot, styles[i], label=config_map[config]["name"])
        for xi, yi in zip(x, y):
            if ymax and (yi > ymax):
                ax.text(xi, ymax - offset, f"{yi:.2f}", ha='center', va='bottom', fontsize=8)
            if ymin and (yi < ymin):
                ax.text(xi, ymin + offset, f"{yi:.2f}",ha='center', va='top', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    if(xtitle):
        ax.set_xlabel(xtitle)
    if show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), columnspacing=1, ncol=len(sub_paths), frameon=True)

def plot_base_mps(data, ylabel, attr, ax=None, xtitle=None, show_legend=True):
    models = list(data.keys())
    configs = list(next(iter(data.values())).keys())
    values = np.array([
        [data[model][config][attr] / data[model]["mps"][attr] for config in configs]
        for model in models
    ])
    plot_bar_chart(models, configs, values, ylabel, ax, xtitle, show_legend=show_legend)

def plot_base_utils(data, ylabel, attr, ax=None, xtitle=None, show_legend=True):
    models = list(data.keys())
    configs = list(next(iter(data.values())).keys())
    values = np.array([
        [data[model][config][attr] / data[model]['utils'][attr] for config in configs]
        for model in models
    ])
    plot_bar_chart(models, configs, values, ylabel, ax, xtitle, show_legend=show_legend)

def plot_normal(data, ylabel, attr, ax=None, xtitle=None, show_legend=True):
    models = list(data.keys())
    configs = list(next(iter(data.values())).keys())
    values = np.array([
        [data[model][config][attr] for config in configs]
        for model in models
    ])
    plot_bar_chart(models, configs, values, ylabel, ax, xtitle, show_legend=show_legend)

def plot_normal_line(data, ylabel, attr, ax=None, xtitle=None, show_legend=True):
    models = list(data.keys())
    configs = list(next(iter(data.values())).keys())
    values = np.array([
        [data[model][config][attr] for config in configs]
        for model in models
    ])
    ymax = max(data[model][config][attr] for model in models for config in configs if config != "time")
    ymin = min(data[model][config][attr] for model in models for config in configs if config != "time")
    margin = 0.1 * (ymax - ymin)

    plot_line_chart(models, configs, values, ylabel, ax, xtitle, ymax + margin, ymin - margin, show_legend=show_legend)


def _add_figure_legend(fig, axes):
    """Build a single figure-level legend that combines bar patches and line markers.

    For series that appear in both a bar subplot and a line subplot (e.g. MPS,
    Time Slicing, batch, context), each legend entry shows the bar patch and the
    line marker side-by-side via HandlerTuple.  Series that only appear in the
    line subplots (e.g. LLM only) are shown as a plain line marker.
    """
    # Collect handles keyed by label, preserving insertion order
    bar_map = {}   # label -> bar handle  (from first bar axes found)
    line_map = {}  # label -> line handle (from first line axes found, preserving series order)

    for ax in axes:
        h_list, l_list = ax.get_legend_handles_labels()
        for h, l in zip(h_list, l_list):
            if isinstance(h, BarContainer):
                if l not in bar_map:
                    bar_map[l] = h
            elif isinstance(h, MplLine2D):
                if l not in line_map:
                    line_map[l] = h

    # Build combined entries in the order they appear in the line charts
    # (which contains all series, including LLM only)
    combined_handles = []
    combined_labels = []
    handler_map = {}

    for label in line_map:
        if label in bar_map:
            pair = (bar_map[label], line_map[label])
            combined_handles.append(pair)
            handler_map[pair] = HandlerBarLine()
        else:
            combined_handles.append(line_map[label])
        combined_labels.append(label)

    fig.legend(combined_handles, combined_labels,
               handler_map=handler_map,
               loc='center', ncol=len(combined_labels),
               bbox_to_anchor=(0.575, 0.97), frameon=True, columnspacing=1,
               handleheight=2.4, handlelength=2.5, borderpad=0.6)


def find_best_mps(trace_tag, model, slo_threshold=99.0, min_mps=0, sub_dir="mps", scheduler_name="constant"):
    """Return the highest MPS % for (trace_tag, model) with >= slo_threshold% LLM SLO satisfaction.
    Falls back to the smallest available MPS % if none qualifies.
    MPS percentages below min_mps are ignored, unless no files exist at all — in that case
    min_mps is ignored and all available files are considered."""
    mps_dir = f"{ROOT_FOLDER}/{sub_dir}"
    if not os.path.isdir(mps_dir):
        return None, None

    batch_size = max_batch_size[model]
    pattern = re.compile(
        r'^llm\[' + re.escape(trace_tag) + r'\]\+' + re.escape(model) +
        r'\(m(\d+(?:\.\d+)?)\)-' + re.escape(scheduler_name) + r'\[' + str(batch_size) + r'\]-' +
        re.escape(DIMENSION) + r'-final_summary\.log$'
    )
    all_satisfaction = {}
    for filename in os.listdir(mps_dir):
        m = pattern.match(filename)
        if not m:
            continue
        pct = int(float(m.group(1)))
        try:
            d = read_final_summary(os.path.join(mps_dir, filename))
            all_satisfaction[pct] = d["satisfaction rate"]
        except Exception:
            pass

    # Apply lower limit, but ignore it if no files exist at all
    mps_satisfaction = {pct: sat for pct, sat in all_satisfaction.items() if pct >= min_mps}
    if not mps_satisfaction:
        mps_satisfaction = all_satisfaction

    if not mps_satisfaction:
        # Broader fallback: find smallest pct from any file matching model/trace in dir
        loose = re.compile(
            r'^llm\[' + re.escape(trace_tag) + r'\]\+' + re.escape(model) +
            r'\(m(\d+(?:\.\d+)?)\)-.*-final_summary\.log$'
        )
        pcts = [int(float(m.group(1))) for f in os.listdir(mps_dir) if (m := loose.match(f))]
        if not pcts:
            return None, None
        return min(pcts), None

    qualifying = [pct for pct, sat in mps_satisfaction.items() if sat >= slo_threshold]
    not_qualifying = [pct for pct, sat in mps_satisfaction.items() if sat < slo_threshold]
    best = max(qualifying) if qualifying else min(mps_satisfaction)  # fall back to smallest
    min_not_qualifying = min(not_qualifying) if not_qualifying else None
    return best, min_not_qualifying


def main():
    global MODEL, ROOT_FOLDER, DIMENSION
    trace_lists = {
        "llama7b": {
            "Azure1,500,100,3600,5,0.022":       {"tag": "LLaMA-7B - Azure"},
            "BurstGPT2,200,100,3600,5,1.5": {"tag": "LLaMA-7B - BurstGPT"},
        },
        "qwen14b": {
            "Azure1,2000,100,3600,5,0.045": {"tag": "Qwen-14B - Azure"},
            "BurstGPT2,1000,100,3600,5,1.8": {"tag": "Qwen-14B - BurstGPT"},
        },
    }
    model_list = ["resnet50", "vgg19", "unet3d", "deeplabv3"]
    sub_paths = {
        "mps":     "mps",
        # "time":    "time",
        # "batch":   "batch",
        "pmps": "pmps",
        "contextminus": "contextminus",
        "context": "context",
    }

    for MODEL in MODELS:
        ROOT_FOLDER = f"../data_final/{MODEL}"
        DIMENSION = MODEL_DIMENSION[MODEL]
        print(f"Generating report for {MODEL}")

        trace_list = trace_lists[MODEL]
        data = {}
        n_traces = len(trace_list)
        fig, all_axes = plt.subplots(n_traces, 2, figsize=(10, 3 * n_traces))
        if n_traces == 1:
            all_axes = [all_axes]  # ensure 2-D indexing works

        for row_idx, (trace, meta) in enumerate(trace_list.items()):
            tag = meta["tag"]
            axes = all_axes[row_idx]
            data[trace] = {}
            for model in model_list:
                data[trace][model] = {}
                best_mps, min_not_qualifying = find_best_mps(trace, model, min_mps=MPS_LOWER_LIMIT)
                print(f"[mps] {tag}/{model}: selected mps={best_mps}, min non-qualifying mps={min_not_qualifying}")
                best_pmps, min_not_qualifying_pmps = find_best_mps(trace, model, min_mps=MPS_LOWER_LIMIT, sub_dir="pmps", scheduler_name="pconstant")
                print(f"[pmps] {tag}/{model}: selected mps={best_pmps}, min non-qualifying mps={min_not_qualifying_pmps}")
                for config, sub_path in sub_paths.items():
                    if config == "mps":
                        d = get_data(model, trace, config, sub_path, best_mps)
                    elif config == "pmps":
                        d = get_data(model, trace, config, sub_path, best_pmps)
                    else:
                        d = get_data(model, trace, config, sub_path)
                    data[trace][model][config] = d
            plot_base_mps(data[trace], 'Normalized Opp. WL Throughput', 'absolute throughput', axes[0], show_legend=False)
            plot_normal_line(data[trace], 'SLO Attainment Rate (%)', 'satisfaction rate', axes[1], show_legend=False)

        _add_figure_legend(fig, all_axes[0])
        fig.tight_layout(rect=[0.15, 0, 1, 0.94])
        fig.subplots_adjust(hspace=0.15)

        # Add row labels on the left side of each row
        for row_idx, (trace, meta) in enumerate(trace_list.items()):
            axes = all_axes[row_idx]
            label = meta["tag"]
            row_mid_y = (min(ax.get_position().y0 for ax in axes) + max(ax.get_position().y1 for ax in axes)) / 2
            left_x = axes[0].get_position().x0 - 0.08
            fig.text(left_x, row_mid_y, label, ha='center', va='center', fontsize=12, rotation=90)

        plt.savefig(f'{OUTPUT_FOLDER}/{MODEL}_ablation.pdf', bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    main()
