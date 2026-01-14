import os

PLOTTING_BIN_SIZE = float(os.environ.get('GVLLM_PLOTTING_BIN_SIZE', '0.5'))

color_list = ["#4B0082", "#377eb8", "#FF4500"]
line_style_list = ["-", "--", ":"]

"""
    Input: List of data
    Return a list where each item is the average of every `chunk_size` items.
"""
def average_chunks(data, chunk_size):
    return [
        sum(data[i : i + chunk_size]) / len(data[i : i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ]

"""
    Read data from files with format:
        metric_name_0 data_0
        metric_name_1 data_1
        ...
    Return a dict containing data in files.
    Used for reading summary files.
"""
def get_data(file_name):
    data = {}
    with open(file_name, 'r') as log_file:
        lines = log_file.readlines()
        for line in lines:
            metric_name, metric = line.strip('\n').split(' ')
            data[metric_name] = float(metric)
    return data