import os
import sys
import time
import logging


from nao_search import epd

from nao_search.common.logger import LoggingConfig

from nao_search.common.utils import random_sequences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import get_top_n


LoggingConfig.Use(filename='nao_arch_search.log', output_to_file=True, level='DEBUG')


# === read sequences and scores from file ===
seqs = []
scores = []

with open('data/encoder.train.input', 'r') as f:
    for line in f:
        seqs.append([int(num) for num in line.split()])


with open('data/encoder.test.input', 'r') as f:
    for line in f:
        seqs.append([int(num) for num in line.split()])

with open('data/encoder.train.target', 'r') as f:
    for line in f:
        scores.append(float(line))

with open('data/encoder.test.target', 'r') as f:
    for line in f:
        scores.append(float(line))



# create model
epd_model = epd.BaseModel(source_length=60,
                          encoder_vocab_size=21,
                          decoder_vocab_size=21,
                          learning_rate=0.0003,
                          num_cpu=1,
                          tensorboard_log='nao_logs',
                          input_processing=False,
                          full_tensorboard_log=False)

# learn
epd_model.learn(X=seqs,
                y=scores,
                epochs=1000,
                log_interval=1,
                tb_log_name='arch_search')


# evaluate model
epd_model.eval(X=seqs,
               y=scores)

# save model
epd_model.save('nao_arch_search.model')


# delete model
del epd_model
epd_model = None



# load model
epd_model = epd.BaseModel.load('nao_arch_search.model',
                               num_cpu=1,
                               tensorboard_log='nao_logs',
                               input_processing=False,
                               full_tensorboard_log=False)

# evaluate model
epd_model.eval(X=seqs,
               y=scores)




