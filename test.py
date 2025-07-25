from profiling.profiler import ProfileServer
from profiling.options import parse_profile_options
import torch.multiprocessing as mp
from plotting.trace_plot import *
import numpy as np
import matplotlib.pyplot as plt
from trace.trace_utils import get_burstgpt_trace_tag, set_burstgpt_trace_config, read_burstgpt_trace_config_json
import pandas as pd
import sys

def test(average_tbt_array):
    tmp_array = []
    answer_array = []
    cnt = 0
    for number in average_tbt_array:
        if(len(tmp_array) == 10):
            tmp_array = sorted(tmp_array)
            answer = min(min(tmp_array[9] - tmp_array[3], tmp_array[8] - tmp_array[2]), min(tmp_array[6], tmp_array[7] - tmp_array[1]))
            answer_array.append(answer)
            if(answer < 0.035):
                cnt += 1
            tmp_array = []
        if(number > 0.000001):
            tmp_array.append(number)
    draw(answer_array, 'histo_.png')
    print(str(cnt) + '/' + str(len(answer_array)), (cnt / len(answer_array)) * 100)

def test2(average_tbt_array):
    tmp_array = []
    answer_array = []
    cnt = 0
    margin = 0.00501
    for id, number in enumerate(average_tbt_array):
        if((id % 50) == 1 and id > 1):
            tmp_array = sorted(tmp_array)
            median = tmp_array[len(tmp_array) // 2]
            in_range = 0
            for x in tmp_array:
                if(x < median + margin and x > median - margin):
                    in_range += 1
            answer_array.append(in_range / len(tmp_array))
            if(in_range / len(tmp_array) > 0.799999):
                cnt += 1
            tmp_array = []
        if(number > 0.000001):
            tmp_array.append(number)
    draw(answer_array, 'histo_.png')
    print(answer_array)
    print(str(cnt) + '/' + str(len(answer_array)), (cnt / len(answer_array)) * 100)

def draw(data, png_filename):
    plt.clf()
    counts, bins, patches = plt.hist(data, bins=12, edgecolor='black', alpha=0.7)
    # Add count labels on top of bars
    for count, bin_edge in zip(counts, bins):
        if count > 0:
            plt.text(bin_edge, count, str(int(count)), ha='left', va='bottom', fontsize=8)
    # Final touches
    plt.title('Histogram of Filtered Average TBT (s)')
    plt.xlabel('Average TBT (s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_filename)

def get_interval_distribution(df, png_filename):
    while(True):
        input_str = input("Input time range:")
        t1, t2 = map(float, input_str.split(' ')) 
        filtered_data = df[(df['Bin Timestamp (s)'] >= t1) & (df['Bin Timestamp (s)'] <= t2)]
        average_tbt_array = filtered_data['Average TBT (s)'].to_numpy()
        average_tbt_array = average_tbt_array[average_tbt_array >= 0.0000001]
        # average_tbt_array = average_tbt_array[average_tbt_array <= 0.1]
        average_tbt_array = np.round(average_tbt_array, 3)
        print(average_tbt_array)
        draw(average_tbt_array, png_filename)
        

if __name__=="__main__":
    # mp.set_start_method('spawn')
    # profile_config = parse_profile_options()
    # profile_server = ProfileServer(profile_config)
    # profile_server.run()
    # Generate random data from a normal distribution


    # Load your CSV file
    config_id =  int(sys.argv[1])
    set_burstgpt_trace_config(config_id)
    trace_config = read_burstgpt_trace_config_json("trace_config.json")
    df = pd.read_csv('./data/llm[' + get_burstgpt_trace_tag() + ']-ttsummary.csv')
    average_tbt_array = df['Average TBT (s)'].to_numpy()
    if("time" in trace_config):
        print("Total time", trace_config["time"])
    if(len(sys.argv) > 2):
        png_filename = sys.argv[2]
    else:
        png_filename = 'histo.png'
    # test(average_tbt_array)
    # test2(average_tbt_array)
    get_interval_distribution(df, png_filename)