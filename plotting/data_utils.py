import sys

def get_data(file_name):
    data = {}
    with open(file_name, 'r') as log_file:
        lines = log_file.readlines()
        for line in lines:
            metric_name, metric = line.strip('\n').split(' ')
            data[metric_name] = float(metric)
    return data

# number per second
def get_throughput(file_name):
    total_time = 0
    with open(file_name, 'r') as log_file:
        lines = log_file.readlines()
        for line in lines:
            total_time += float(line.strip('\n'))
        total_executed = len(lines)
    return total_executed / (total_time / 1000)