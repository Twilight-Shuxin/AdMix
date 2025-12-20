import torch
import matplotlib.pyplot as plt
import numpy as np

def average_chunks(data, chunk_size):
    """Return a list where each item is the average of every `chunk_size` items."""
    return [
        (data[i][0], sum(y for _, y in data[i:i + chunk_size]) / len(data[i:i + chunk_size]))
        for i in range(0, len(data), chunk_size)
    ]

# Accepting data1 & data2 as numpy ndarrays
def plot(timestamps, real, predicted, file_name):
    # Plotting
    chunk_size = 20
    plt.figure(figsize=(10, 5))
    real_data = list(zip(timestamps, real))
    predicted_data = list(zip(timestamps, predicted))
    real_data = average_chunks(real_data, chunk_size)
    predicted_data = average_chunks(predicted_data, chunk_size)
    plt.plot([t[0] for t in real_data], [t[1] for t in real_data], label='True', color='blue', alpha=0.7)
    plt.plot([t[0] for t in predicted_data], [t[1] for t in predicted_data], label='Predicted', color='red', alpha=0.7)
    print(len(predicted_data))
    plt.xlabel('Time')
    # plt.ylabel(file_name + '_throughput')
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

def plot_distribution(data, name):
    plt.figure()
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {name}')
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.savefig(name + ".png")

def plot_distributions(X, y1, y2, headers, tag):
    cols = [
        't1 prefill intensity',
        't1 prefill average length',
        't1 decode intensity',
        't1 decode average length'
    ]
    col_indices = [headers.index(col) for col in cols]
    for id, col in enumerate(cols):
        plot_distribution(X[:,col_indices[id]], "-".join([tag, col]))
    plot_distribution(y1, "-".join([tag, 'prefill throughput']))
    plot_distribution(y2, "-".join([tag, 'decode throughput']))
