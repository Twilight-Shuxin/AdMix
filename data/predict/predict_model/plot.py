import torch
import matplotlib.pyplot as plt
import numpy as np

# Accepting data1 & data2 as numpy ndarrays
def plot(timestamps, real, predicted, file_name):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, real, label='true', color='blue')
    plt.plot(timestamps, predicted, label='predicted', color='red')
    plt.xlabel('time')
    plt.ylabel(file_name + '_throughput')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.close() 

def plot_abs_error_cdf(real, predicted, file_name):
    """
    Plot CDF of Absolute Error for latency prediction.
    """
    abs_error = np.abs(predicted - real)
    abs_error_sorted = np.sort(abs_error)
    for i, error in enumerate(abs_error):
        if(error>100):
            print("error great in: ", i)

    cdf = np.linspace(0, 1, len(abs_error_sorted))
    plt.figure(figsize=(5, 4))

    plt.plot(abs_error_sorted, cdf, '.')
    plt.xlabel("Absolute Error (s)")
    plt.ylabel("CDF")
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.close() 

def plot_abs_pct_error_cdf(real, predicted, file_name):
    """
    Plot CDF of Absolute Percentage Error for latency prediction.
    """
    abs_error = np.abs(predicted - real)
    abs_pct_error = abs_error / real
    abs_pct_error_sorted = np.sort(abs_pct_error)
    cdf = np.linspace(0, 1, len(abs_pct_error_sorted))

    plt.figure(figsize=(5, 4))
    plt.plot(abs_pct_error_sorted, cdf, '.')
    plt.xlabel("Absolute Percentage Error")
    plt.ylabel("CDF")
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_name + ".png")
    plt.close() 