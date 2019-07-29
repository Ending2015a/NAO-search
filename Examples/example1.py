import os
import sys
import time
import logging


from nao_search import epd

from nao_search.common.logger import LoggingConfig

from nao_search.common.utils import random_sequences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import get_top_n


LoggingConfig.Use(filename='nao_training.log', output_to_file=True, level='DEBUG')


# === generate sequences ===

seqs = random_sequences(length=60,
                        seq_num=300,
                        vocab_size=20)

# === calculate score ===

def get_score(seq):
    return sum(seq)

scores = []
for seq in seqs:
    scores.append(get_score(seq))

norm_scores = min_max_normalization(scores)

# create model
epd_model = epd.BaseModel(source_length=60,
                          encoder_vocab_size=20,
                          decoder_vocab_size=20,
                          num_cpu=1,
                          tensorboard_log='epd_log',
                          full_tensorboard_log=False)

epd_model.learn(X=seqs,
                y=norm_scores,
                log_interval=1,
                tb_log_name='rndseq_search')


epd_model.save('epd_seq_search.model')

