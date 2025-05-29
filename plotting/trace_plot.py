import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from plotting.constants import *

def recover_intervals(start_times, intervals, max_end_time):
    existing_starts = {start for start, end in intervals}
    merged_intervals = intervals[:]
    last_start = start_times[0]
    recorded_id = 0
    for id, start in enumerate(start_times):
        if(recorded_id >= len(existing_starts) or math.fabs(start - existing_starts[recorded_id]) > 1e-16):
            merged_intervals.apend(start, max_end_time)
        else:
            recorded_id += 1
    merged_intervals.sort()
    return merged_intervals

def get_bin_from_intervals(time_intervals):
    # Extract the maximum timestamp from the end times
    max_time = max(end for _, end in time_intervals)
    time_period = BIN_SIZE
    bins = np.arange(0, max_time + time_period, time_period)

    # Create an array to store intensity values
    intensity = np.zeros(len(bins) - 1)

    # Iterate over each time interval
    for start, end in time_intervals:
        active_bins = np.digitize(np.arange(start, end, time_period), bins) - 1
        for idx in active_bins:
            if 0 <= idx < len(intensity):
                intensity[idx] += 1
    
    # Calculate midpoints of bins for intensity plotting
    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    return (bin_midpoints, intensity)

def trace_plot_aggregated_intensity_with_analysis_data(aggregated_intensity, analysis_data, y_tag, plot_file):
    bin_midpoints, intensity = aggregated_intensity
    # Extract latency timestamps and values
    analysis_data_timestamps = [entry[0] for entry in analysis_data]
    analysis_data_values = [entry[1] for entry in analysis_data]

    # Create figure and plot intensity
    fig, ax1 = plt.subplots(figsize=(10, 5))

    y_tag_name, y_tag_unit = y_tag
    ax1.plot(bin_midpoints, intensity, linestyle='-', color='b', label='Intensity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Intensity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    # Create secondary y-axis for latency
    ax2 = ax1.twinx()
    ax2.plot(analysis_data_timestamps, analysis_data_values, linestyle='--', color='r', label='Latency')
    ax2.set_ylabel((y_tag_name + ' ' + y_tag_unit) if(len(y_tag_unit) > 0) else y_tag_name, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Title and layout
    plt.title('Aggregated Request Intensity & ' + y_tag_name + ' Over Time')
    fig.tight_layout()
    plt.savefig(plot_file)


def trace_plot_aggregated_intensity(aggregated_intensity, plot_file):
    bin_midpoints, intensity = aggregated_intensity
    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.plot(bin_midpoints, intensity, linestyle='-', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title('Aggregated Request Intensity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file)

def bin_data(data, bin_size=BIN_SIZE):
    """
    Groups data into bins of `bin_size` seconds and computes the average.
    
    :param data: List of (timestamp, value) pairs.
    :param bin_size: Size of each bin in seconds.
    :return: Binned (timestamp, value) pairs.
    """
    timestamps, values = zip(*data)
    min_time, max_time = min(timestamps), max(timestamps)
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    
    binned_values = []
    for i in range(len(bins) - 1):
        bin_start, bin_end = bins[i], bins[i + 1]
        bin_items = [v for t, v in data if bin_start <= t < bin_end]
        avg_value = np.mean(bin_items) if bin_items else np.nan  # Handle empty bins
        binned_values.append((bin_start, avg_value))

    return [(t, v) for t, v in binned_values if not np.isnan(v)]  # Remove NaNs

def trace_plot_intensity_with_analysis_data(intensity_data_all, intensity_data_first_received, analysis_data, plot_file):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bin intensity data
    binned_all = bin_data(intensity_data_all)
    binned_first = bin_data(intensity_data_first_received)

    # Extract timestamps and values
    timestamps_all, values_all = zip(*binned_all)
    timestamps_first, values_first = zip(*binned_first)

    # Plot smoothed intensity data on left y-axis
    ax1.plot(timestamps_all, values_all, label="All", color="#4B0082", linestyle="-")
    ax1.plot(timestamps_first, values_first, label="Waiting First", color="#377eb8", linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Intensity")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Create right y-axis for analysis data
    ax2 = ax1.twinx()
    timestamps_analysis, values_analysis = zip(*analysis_data)
    ax2.plot(timestamps_analysis, values_analysis, label="Analysis Data", color="#FF4500", linestyle=":")
    ax2.set_ylabel("Latency (s)")
    ax2.legend(loc="upper right")

    # Save plot
    plt.title("Request Intensity & Latency")
    plt.savefig(plot_file)
    plt.show()

def trace_plot_request_intensity(timestamps, plot_file):
    # Convert timestamps to NumPy array for efficient processing
    SUB_BIN_SIZE = 1
    timestamps = np.array(timestamps)

    # Define bin edges from min to max timestamp
    min_time, max_time = np.min(timestamps), np.max(timestamps)
    bins = np.arange(min_time, max_time + SUB_BIN_SIZE, SUB_BIN_SIZE)

    # Count occurrences per bin
    intensity, _ = np.histogram(timestamps, bins=bins)

    # Compute bin centers for plotting
    bin_centers = bins[:-1] + (SUB_BIN_SIZE / 2)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, intensity, linestyle="-", color="blue", label="Request Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Request Count")
    plt.title("Request Intensity Over Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(plot_file)

