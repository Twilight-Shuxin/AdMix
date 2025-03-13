from data_utils import *
import sys
import matplotlib.pyplot as plt
import numpy as np

# usage: python plot_single_curve.py data_mps vgg19

if __name__=="__main__":
    model_name = sys.argv[2]
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    for batch_size in batch_sizes:
        values_mean = []
        values_99 = []
        for percentage in percentages:
            if(percentage != 100):
                tag = "_".join([model_name + "(" + str(percentage) + ")", str(batch_size), "summary.log"])
            else:
                tag = "_".join([model_name, str(batch_size), "summary.log"])
            data = get_data("../" + sys.argv[1] + "/" + tag)
            values_mean.append(data["mean"])
            values_99.append(data["99%"])
        plt.plot(percentages, values_mean, 'b', label = 'mean')
        plt.plot(percentages, values_99, 'orange', label = '99%')

        # Add labels and title
        plt.xlabel('mps percentile (%)')
        plt.ylabel('latency (ms)')
        plt.title('latency change of ' + model_name + ' of batch size ' + str(batch_size) + ' with mps percentage')

        # Add a legend
        plt.legend()
        plt.savefig(model_name + "_" + str(batch_size) + ".pdf")
        plt.clf()
        # plt.savefig("../" + sys.argv[1] + "/" + model_name + "_" + str(batch_size) ".pdf")