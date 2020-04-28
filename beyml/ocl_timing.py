import pyopencl as cl  # Import the OpenCL GPU computing API
import numpy as np  # Import number tools
from time import time  # Import time tools
import kernelvis


import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def init_gpu_test():
    context = cl.create_some_context() 
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    

    a = np.random.rand(10 * 1024 * 1024).astype(np.float32)  # Create a random array to add
    b = np.random.rand(10 * 1024 * 1024).astype(np.float32)  # Create a random array to add

    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)
    
    program = cl.Program(context, """
    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
    {
        int i = get_global_id(0);
        int j = get_global_id(1);
        int pitch = get_global_size(0);
        int idx = j * pitch + i;

        if (idx < 1024 * 1024 * 10)
        {
            c[idx] = a[idx] + b[idx];
        }
        
    }""").build()  # Compile the device program

    sum_kernel = program.sum
    sum_kernel.set_args(a_buffer, b_buffer, c_buffer)
    test_func = lambda local_size, global_size : cl.enqueue_nd_range_kernel(queue, sum_kernel, global_size, local_size)

    local_sizes = np.array([[8,8], [16, 16], [256, 1]])
    global_sizes = np.array([[1024,1024],[1024 * 5, 1024 * 5],  [1024 * 10, 1024 * 10]])


    analysis = kernelvis.measure_kernel_on_different_work_group_sizes(local_sizes, global_sizes, test_func)

    print(analysis)

    print(analysis)


init_gpu_test()




#def gpu_array_sum(a, b):
#    context = cl.create_some_context()  # Initialize the Context
#    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Instantiate a Queue with profiling (timing) enabled
#    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
#    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
#    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)  # Create three buffers (plans for areas of memory on the device)
#    program = cl.Program(context, """
#    __kernel void sum(__global const float *a, __global const float *b, __global float *c)
#    {
#        int i = get_global_id(0);
#        int j;
#        for(j = 0; j < 1000; j++)
#        {
#            c[i] = a[i] + b[i];
#        }
#    }""").build()  # Compile the device program
#    gpu_start_time = time()  # Get the GPU start time
#    event = program.sum(queue, a.shape, None, a_buffer, b_buffer, c_buffer)  # Enqueue the GPU sum program XXX
#    event.wait()  # Wait until the event finishes XXX
#    elapsed = 1e-9*(event.profile.end - event.profile.start)  # Calculate the time it took to execute the kernel
#    print("GPU Kernel Time: {0} s".format(elapsed))  # Print the time it took to execute the kernel
#    c_gpu = np.empty_like(a)  # Create an empty array the same size as array a
    
    
#    cl.enqueue_copy(queue, c_buffer, c_gpu)  # Read back the data from GPU memory into array c_gpu
    
#    gpu_end_time = time()  # Get the GPU end time
#    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))  # Print the time the GPU program took, including both memory copies
#    return c_gpu  # Return the sum of the two arrays

#gpu_array_sum(a, b)  # Call the function that sums two arrays on the GPU