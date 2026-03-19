import json
import matplotlib.pyplot as plt
import numpy as np

ROOT_FOLDER = "../data_staged/llama7b"
PROFILE_DATA_FOLDER = "../profile_data"
OUTPUT_FOLDER = "./outputs"
DIMENSION = "d1"

scheduler_map = {
    "batch": "batch",
    "mps": "constant",
    "time": "constant",
    "utils": None
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

map_subpath = {
    "time": "Time Slicing",
    "batch": "batch",
    "mps": "MPS",
    "utils": "LLM only",
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


def get_data(model, trace_tag, sub_path, percentage = None):
    data = {}
    scheduler = scheduler_map[sub_path]
    if(scheduler != None):
        if(scheduler == "constant"):
            scheduler += f"[{max_batch_size[model]}]" if sub_path == "mps" else f"[{time_batch_size[model]}]"
        set_tag = f'llm[{trace_tag}]+{model}'
        if(sub_path == "mps"):
            set_tag += f"(m{percentage})"
        set_tag += "-" + scheduler
        path_prefix = f"llm-{set_tag}"
        offline_path_prefix = f'{model}-{set_tag}'
        data.update(read_final_summary(f'{ROOT_FOLDER}/{sub_path}/{set_tag}-{DIMENSION}-final_summary.log'))
    else:
        path_prefix = "llm[" + trace_tag + "]"
        data.update(read_final_summary(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-final_summary.log', False))
    data.update(read_stats(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-ftsummary.log', "ft"))
    data.update(read_stats(f'{ROOT_FOLDER}/{sub_path}/{path_prefix}-{DIMENSION}-btsummary.log', "bt"))
    return data

def plot_bar_chart(models, sub_paths, values, ylabel, ax = None, xtitle = None, ymax = None, ymin = None): 
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

    hatches = ['///', 'xx', '-', '+']
    x = np.arange(len(models))  # model positions
    bar_width = 0.8 / len(sub_paths)  # divide bar width by number of bars per group
    
    for i, sub_path in enumerate(sub_paths):
        y = values[:, i]
        y_plot = np.clip(y,
            ymin if ymin is not None else -np.inf,
            ymax if ymax is not None else np.inf
        )
        ax.bar(x + i * bar_width, y_plot, hatch=hatches[i], edgecolor='white', width=bar_width, label=map_subpath[sub_path])
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), columnspacing = 1, ncol=len(sub_paths), frameon=True)

def plot_line_chart(models, sub_paths, values, ylabel, ax = None, xtitle = None, ymax = None, ymin = None):
    x = np.arange(len(models))  # model positions
    styles = ['--x', ':s', '-o', '-.^']
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

    # Plot a line for each sub_path
    for i, sub_path in enumerate(sub_paths):
        y = values[:, i]
        y_plot = np.clip(y,
            ymin if ymin is not None else -np.inf,
            ymax if ymax is not None else np.inf
        )
        ax.plot(x, y_plot, styles[i], label=map_subpath[sub_path])
        for xi, yi in zip(x, y):
            if ymax and (yi > ymax):
                ax.text(xi, ymax - offset, f"{yi:.1f}", ha='center', va='bottom', fontsize=8)   
            if ymin and (yi < ymin):
                ax.text(xi, ymin + offset, f"{yi:.1f}",ha='center', va='top', fontsize=8) 

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    if(xtitle):
        ax.set_xlabel(xtitle)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), columnspacing = 1, ncol=len(sub_paths), frameon=True)
    plt.tight_layout()

def plot_base_mps(data, ylabel, attr, ax = None, xtitle = None):
    models = list(data.keys())
    sub_paths = list(next(iter(data.values())).keys())  # get keys from first model
    # Extract throughput matrix
    values = np.array([
        [data[model][sub_path][attr] / data[model]["mps"][attr] for sub_path in sub_paths]
        for model in models
    ])
    plot_bar_chart(models, sub_paths, values, ylabel, ax, xtitle)

def plot_base_utils(data, ylabel, attr, ax = None, xtitle = None):
    models = list(data.keys())
    sub_paths = list(next(iter(data.values())).keys())  # get keys from first model
    # Extract throughput matrix
    values = np.array([
        [data[model][sub_path][attr] / data[model]['utils'][attr] for sub_path in sub_paths]
        for model in models
    ])
    plot_bar_chart(models, sub_paths, values, ylabel, ax, xtitle)

def plot_normal(data, ylabel, attr, ax = None, xtitle = None):
    models = list(data.keys())
    sub_paths = list(next(iter(data.values())).keys())  # get keys from first model
    # Extract throughput matrix
    values = np.array([
        [data[model][sub_path][attr] for sub_path in sub_paths]
        for model in models
    ])
    plot_bar_chart(models, sub_paths, values, ylabel, ax, xtitle)

def plot_normal_line(data, ylabel, attr, ax = None, xtitle = None):
    models = list(data.keys())
    sub_paths = list(next(iter(data.values())).keys())  # get keys from first model
    # Extract throughput matrix
    values = np.array([
        [data[model][sub_path][attr] for sub_path in sub_paths]
        for model in models
    ])
    ymax = max(data[model][sub_path][attr] for model in models for sub_path in sub_paths if sub_path != "time")
    ymin = min(data[model][sub_path][attr] for model in models for sub_path in sub_paths if sub_path != "time")
    margin = 0.1 * (ymax - ymin)
    
    plot_line_chart(models, sub_paths, values, ylabel, ax, xtitle, ymax + margin, ymin - margin)


def main():
    # trace_list = ["BurstGPT1,301,100,3600,2.8", "Azure1,0,100,3600,0.1"]
    trace_list = ["BurstGPT2,200,100,3600,9"]
    tag_list = ["BurstGPT2", "Azure"]
    mps_list = [20, 10]
    mps_list = [10]
    model_list = ["resnet50", "vgg19", "unet3d", "deeplabv3"]
    batch_size_list = [128, 128, 16, 32]
    # sub_paths = ["mps", "time", "batch"]
    sub_paths = ["mps", "batch"]

    data = {}
    for id, trace in enumerate(trace_list):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.subplots_adjust(wspace=-0.1)  
        data[trace] = {}
        for model in model_list:
            data[trace][model] = {}
            for sub_path in sub_paths:
                if(sub_path == "mps"):
                    d = get_data(model, trace, sub_path, mps_list[id])
                else:
                    d = get_data(model, trace, sub_path)
                data[trace][model][sub_path] = d
        plot_base_mps(data[trace], 'Normalized Offline WL Throughput', 'absolute throughput', axes[0])
        d = get_data(model, trace, 'utils')
        for model in model_list:
            data[trace][model]['utils'] = d
        plot_normal_line(data[trace], 'SLO Attainment Rate (%)', 'satisfaction rate', axes[1])
        plot_normal_line(data[trace], 'SM Activity (%)', 'sm_active', axes[2])
        fig.tight_layout()
        plt.savefig(f'{OUTPUT_FOLDER}/{tag_list[id]}.pdf')
        plt.clf()
    
    for id, trace in enumerate(trace_list):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(wspace=-0.1)
        plot_normal_line(data[trace], 'Average TTFT (s)', 'ft_mean', axes[0])
        plot_normal_line(data[trace], 'Average TBT (s)', 'bt_mean', axes[1])
        plt.savefig(f'{OUTPUT_FOLDER}/{tag_list[id]}_latency.pdf')
        plt.clf()

    profile_data = {model: load_profile_data(model) for model in model_list}
    for id, trace in enumerate(trace_list):
        # Compute normalized energy: actual / (llm_only + side_model_separate)
        norm_energy = {}
        for model, batch_size in zip(model_list, batch_size_list):
            llm_energy = data[trace][model]['utils']['energy']
            side_energy_per_output = profile_data[model][str(batch_size)]['avg_energy'] / 1000 # Energy in profile is J
            norm_energy[model] = {}
            for sub_path in sub_paths:
                d = data[trace][model][sub_path]
                separate_energy = llm_energy + d['absolute throughput'] * side_energy_per_output
                norm_energy[model][sub_path] = round((d['energy'] / separate_energy) * 100, 1)

        values = np.array([
            [norm_energy[model][sub_path] for sub_path in sub_paths]
            for model in model_list
        ])
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_line_chart(model_list, sub_paths, values, 'Normalized Energy (%)', ax)
        fig.tight_layout()
        plt.savefig(f'{OUTPUT_FOLDER}/{tag_list[id]}_energy.pdf')
        plt.clf()


if __name__ == "__main__":
    main()
