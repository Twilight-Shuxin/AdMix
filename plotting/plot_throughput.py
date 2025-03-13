import matplotlib.pyplot as plt
import numpy as np
import sys
from data_utils import *

# Sample data

model_name = sys.argv[2]
categories = ['1', '2', '4', '8', '16', '32']
percentages = [x for x in range(10, 101, 10)]


for percentage in percentages:
    fig, ax1 = plt.subplots() 
    throughput_data = []
    percentile_99_data = []
    for batch_size in categories:
        if(percentage != 100):
            tag = "_".join([model_name + "(" + str(percentage) + ")", str(batch_size)])
        else:
            tag = "_".join([model_name, str(batch_size)])
        throughput = get_throughput("../" + sys.argv[1] + "/" + tag + "_all.log")
        throughput_data.append(throughput * int(batch_size))
        percentile_99_data.append(get_data("../" + sys.argv[1] + "/" + tag + "_summary.log")["99%"])
    
    # bar plot
    print(throughput_data)
    bar = ax1.bar(categories, throughput_data, color='tab:blue', label='throughput')
    ax1.set_xlabel('batch_size')
    ax1.set_ylabel('throughput (n/s)', color='tab:blue')
    ax1.tick_params('y', colors='tab:blue')

    # Create a second y-axis for the line plot
    ax2 = ax1.twinx()
    line, = ax2.plot(categories, percentile_99_data, color='tab:orange', marker='o', label='latency')
    ax2.set_ylabel('99 percentile latency (ms)', color='tab:orange')
    ax2.tick_params('y', colors='tab:orange')

# Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + [line], labels + [labels2[0]], loc='upper left')

    plt.title(model_name + ': Throughput & latency with ' + str(percentage) + "% resources")
    plt.savefig(model_name + "(" + str(percentage) + ").pdf")
    # plt.savefig("ok.png")
    plt.clf()
    # exit(0)

