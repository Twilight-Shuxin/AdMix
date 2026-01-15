import torch
import matplotlib.pyplot as plt
import numpy as np

def average_chunks(data, chunk_size):
    """Return a list where each item is the average of every `chunk_size` items."""
    return [
        sum(y for y in data[i:i + chunk_size]) / len(data[i:i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ]

# Accepting data1 & data2 as numpy ndarrays
def plot_predicted(timestamps, real, predicted, file_name):
    # Plotting
    plt.figure(figsize=(10, 5))
    real_data = list(zip(timestamps, real))
    predicted_data = list(zip(timestamps, predicted))

    chunk_size = 20
    steped_timestamps = [timestamp for timestamp in range(0, len(timestamps), chunk_size)]
    real_data = average_chunks(real_data, chunk_size)
    predicted_data = average_chunks(predicted_data, chunk_size)
    plt.plot(steped_timestamps, real_data, label='True', color='blue', alpha=0.7)
    plt.plot(steped_timestamps, predicted_data, label='Predicted', color='red', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Averaged Throughput')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name + ".pdf")
    plt.close() 

def plot_abs_error_cdf(real, predicted, file_name):
    """
    Plot CDF of Absolute Error for latency prediction.
    Downsamples uniformly to reduce the number of plotted points.
    """
    abs_error = np.abs(predicted - real)
    abs_error_sorted = np.sort(abs_error)

    # Print large errors for inspection
    # for i, error in enumerate(abs_error):
    #     if error > 100:
    #         print("error greater in:", i)

    cdf = np.linspace(0, 1, len(abs_error_sorted))

    # --- Downsample for plotting ---
    max_points = 5000
    if len(abs_error_sorted) > max_points:
        idx = np.linspace(0, len(abs_error_sorted) - 1, max_points, dtype=int)
        abs_error_sorted = abs_error_sorted[idx]
        cdf = cdf[idx]

    # --- Plot ---
    plt.figure(figsize=(5, 4))
    plt.plot(abs_error_sorted, cdf, '.', alpha=0.7)
    plt.xlabel("Absolute Error")
    plt.ylabel("CDF")
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_name + ".pdf")
    plt.close()

def plot_abs_pct_error_cdf(real, predicted, file_name):
    """
    Plot CDF of Absolute Percentage Error for latency prediction.
    Currently not used due to small real values (very close to 0 or 0).
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
    plt.savefig(file_name + ".pdf")
    plt.close() 