import matplotlib.pyplot as plt
import numpy as np
import sys
from data_utils import *

# python plot_interference.py data data_mps resnet50,vgg19 4,4

if __name__=="__main__":
    model_names = sys.argv[3].strip("\n").split(",")
    batch_sizes = sys.argv[4].strip("\n").split(",")
    percentages = [40, 40, 50, 50, 60, 60, 70, 80, 100]
    percentages_r = [80, 60, 70, 50, 60, 40, 50, 40, 100]
    cl = ['tab:blue', 'orange']
    cl2 = ['blue', 'tab:orange']

    categories = [
        "(" + str(percentages[i]) + "," + str(percentages_r[i])  + ")" for i in range(len(percentages))
    ]
    print(categories)
    values_mean = {}
    values_min = {}
    values_99 = {}
    values_99_single = {}
    lower_error = {}
    upper_error = {}
    error = {}
    
    for model_name in model_names:
        values_mean[model_name] = []
        values_min[model_name] = []
        values_99[model_name] = []
        values_99_single[model_name] = []
        lower_error[model_name] = []
        upper_error[model_name] = []
        error[model_name] = []

    for id_i, percentage in enumerate(percentages):
        percentage_l = (percentage, percentages_r[id_i])
        tag = ""
        for id_j, model_name in enumerate(model_names):
            tag += model_name + "(" + str(batch_sizes[id_j]) + "_" + str(percentage_l[id_j]) + ")"
        for id_j, model_name in enumerate(model_names):
            data = get_data("../" + sys.argv[1] + "/" + model_name + "_" + str(batch_sizes[id_j]) + "_" + tag + "_summary.log")
            values_mean[model_name].append(data["mean"])
            values_min[model_name].append(data["min"])
            values_99[model_name].append(data["99%"])
        for id_j, model_name in enumerate(model_names):
            if(percentage_l[id_j] != 100):
                tag = "_".join([model_name + "(" + str(percentage_l[id_j]) + ")", str(batch_sizes[id_j]), "summary.log"])
            else:
                tag = "_".join([model_name, str(batch_sizes[id_j]), "summary.log"])
            data = get_data("../" + sys.argv[2] + "/" + tag)
            values_99_single[model_name].append(data["99%"])

    for model_name in model_names:
        lower_error[model_name] = np.array(values_mean[model_name]) - np.array(values_min[model_name])
        upper_error[model_name] = np.array(values_99[model_name]) - np.array(values_mean[model_name])
        error[model_name] = [lower_error[model_name], upper_error[model_name]]

    x = np.arange(len(categories))  # The label locations
    width = 0.35  # The width of the bars
    fig, ax = plt.subplots(figsize=(15, 8))
    for id, model_name in enumerate(model_names):
        bars = ax.bar(x - (width/2 if id == 0 else -width/2), values_mean[model_name], width, yerr=error[model_name], capsize=5, label = model_name, color=cl[id], alpha=0.8)
        ax.plot(x - (width/2 if id == 0 else -width/2), values_99_single[model_name], color=cl2[id], marker='o', linestyle = '--', label= model_name + "_mps_99")

    # Add labels and title
    ax.set_xlabel('MPS settings')
    ax.set_ylabel('latency (ms)')
    tag = ""
    for i, model_name in enumerate(model_names):
        tag += model_name + "_(b" + str(batch_sizes[i] + ")")
    ax.set_title(tag + ': Mean, Min, and 99 percentile latency')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mps_comb/" + tag + ".pdf")

