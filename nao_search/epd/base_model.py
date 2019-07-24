from abc import ABC, abstractmethod

import os
import sys
import time
import logging

import tensorflow as tf

from collections import OrderedDict

from nao_search.common.utils

class BaseModel(ABC):
    def __init__(self):
        pass

    def setup_model

    def get_parameter_list(self):
        pass

    def get_parameters(self):
        parameters = self.get_parameter_list()
        parameter_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_value))

        return return_dictionary



