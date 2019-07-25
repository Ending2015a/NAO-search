import os
import sys
import time
import logging
import multiprocessing

import tensorflow as tf


def make_session(num_cpu=None, make_default=False, graph=None):
    if num_cpu is None or num_cpu <= 0:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))

    tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)

    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)



__all__ = [
        make_session.__name__
        ]
