#Copyright (c) <year> <copyright holders>

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

"""
Entry File, which contains the main method.
"""
import os
import time
import multiprocessing
import math

from pathlib import Path
import tensorflow as tf

from Controller_Component.Controller import Controller

def get_cpu_quota_within_docker():
    cpu_cores = None

    cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        # we are in a linux container with cpu quotas!
        with cfs_period.open('rb') as p, cfs_quota.open('rb') as q:
            p, q = int(p.read()), int(q.read())

            # get the cores allocated by dividing the quota
            # in microseconds by the period in microseconds
            cpu_cores = math.ceil(q / p) if q > 0 and p > 0 else None

    return cpu_cores

def main():
    """
    Main method which initialises and starts the execution via the controller.
    The type of the execution specified in the controllerConfig.

    This function prints information about soft- and hardware as well.
    """
    ###### Print information ######
    cpu_cores = get_cpu_quota_within_docker() or multiprocessing.cpu_count()

    print("Main: Running Tensorflow version: " + str(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Build with Cuda: ", tf.test.is_built_with_cuda())
    print("Num CPU-Cores Available: ", cpu_cores)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
    print("inter_op_parallelism_threads: ", tf.config.threading.get_inter_op_parallelism_threads())
    print("intra_op_parallelism_threads: ", tf.config.threading.get_intra_op_parallelism_threads())
    print("##########################################")
    print(tf.config.list_physical_devices('GPU'))
        
    print("Main: Starting initialisation ...")
    start_initialisation_time = time.time()

    # Seems to be a error in the precompiled TensorFlow, set memory_growth=True in session config as a workaround.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
 
    
    ###### Execution of the program ######
    config_path = os.path.dirname(os.path.realpath(__file__)) + "/controllerConfig.json"
    controller = Controller(config_path)

    initialisation_ok = controller.init()
    end_initialisation_time = time.time()
    print("#########FINSIHED INITIALISATION##########")
    print("Initialisation successful: ", initialisation_ok)
    print("Time for initialisation: ", end_initialisation_time-start_initialisation_time, "s")
    print("##########################################")

    print("Main: Starting execution ...")
    start_execution_time = time.time()
    execution_ok = controller.execute()
    end_execution_time = time.time()
    print("############FINSIHED EXECUTION############")
    print("Execution successful: ", execution_ok)
    print("Time for execution: ", end_execution_time-start_execution_time, "s")
    print("##########################################")

main()
