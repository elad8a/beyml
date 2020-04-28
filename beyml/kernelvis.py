import pandas as pd
import pyopencl as cl
import numpy as np
from random import random

def launch_and_measure_avarage(local_size, global_size, launch_count=10):
    total_duration = 0.0
    for i in range(launch_count):
        duration = abs(random()) * (global_size[0] * global_size[1])

        total_duration += duration

    return total_duration / launch_count

def wait_end_measure_ms(event):
    event.wait()
    time_ns = event.profile.end - event.profile.start;
    #time_s = time_ns * 1e-9
    #time_ms = time_s * 10e3
    return time_ns * 1e-6


def measure_kernel_on_different_work_group_sizes(local_sizes, global_sizes, kernel_launcher, launch_count=10):
    measures_columns = {}
     
    for local_size in local_sizes:
        column = []
        for global_size in global_sizes:
            events = []
            for i in range(launch_count):
                event = kernel_launcher(local_size, global_size)
                events.append(event)
            average_duration = np.average(np.array([wait_end_measure_ms(ev) for ev in events]))
            column.append(average_duration)
        measures_columns[str(local_size)] = column

    rows = [str(global_size) for global_size in global_sizes]
    dataframe = pd.DataFrame(measures_columns, index=rows)

    return dataframe


def test():
    local_sizes = np.array([[8,8], [16, 16], [32, 32]])
    global_sizes = local_sizes * 100
    data = measure_kernel_on_different_work_group_sizes(local_sizes, global_sizes)

    print('time: global_size / local_size')
    print(data)

#test()