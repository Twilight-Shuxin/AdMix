from plotting.trace_plot import *
from trace.trace_utils import BurstGPTTrace
from profiling.constants import *

DATA_PREFIX = "/home/k43015/MPSTest/data/"

# with open('./data_llm(100,501,15000,3.7)/llm(n)-all.log') as input_file:
#     lines = input_file.readlines()
#     timestamps = [float(line.strip('\n').split(' ')[0]) for line in lines]

# trace_plot_request_intensity(timestamps, '/home/k43015/MPSTest/data_llm(100,501,15000,3.7)/a_request_intensity.png')

# trace = BurstGPTTrace('BurstGPT_1.csv')
# with open('/home/k43015/MPSTest/data/request_intensity.txt', 'w') as log_file:
#     while(True):
#         req_time, sample_in_length, sample_out_length = trace.next()
#         log_file.write(str(req_time) + '\n')
#         if(req_time > 300):
#             break

# with open(DATA_PREFIX + 'request_intensity.txt') as input_file:
#     lines = input_file.readlines()
#     timestamps = [float(line.strip('\n')) for line in lines]

# trace_plot_request_intensity(timestamps, DATA_PREFIX + 'a_request_intensity.png')


intensity = []
bin_midpoints = []
with open(DATA_PREFIX + 'llm(n)-aggregated_intensity_all.log') as input_file:
    lines = input_file.readlines()
    for line in lines:
        bin_midpoint_data, intensity_data = map(float, line.strip('\n').split(' '))
        bin_midpoints.append(bin_midpoint_data)
        intensity.append(intensity_data)

values = get_dcgm_data(DATA_PREFIX + 'llm(n)-sm_active.log')
values = [[index + 1, value] for index, value in enumerate(values)]
print(len(values))
trace_plot_aggregated_intensity_with_analysis_data((bin_midpoints, intensity), values, ('SM active', ''), 
    DATA_PREFIX + 'aggregated_intensity_sm_active.png')
