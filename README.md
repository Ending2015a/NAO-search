# NAO Skill Search

## Introduction
A simplified tensorflow version of Neural Architecture Optimization.

## Example: Learning Random Generated Sequences
```python
import os
import sys
import time
import logging

from nao_search import epd

from nao_search.common.logger import LoggingConfig

from nao_search.common.utils import random_sequences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import get_top_n


# === Generate sequences ===

seqs = random_sequences(length=60,
                        seq_num=300,
                        vocab_size=20)


# === Calculate scores ===

def get_score(seq):
    return sum(seq)

scores = []
for seq in seqs:
    scores.append(get_score(seq))

norm_scores = min_max_normalization(scores)

# === Create model ===

epd_model = epd.BaseModel(source_length=60,
                          encoder_vocab_size=21,        # vocab_size + <SOS> = 21
                          decoder_vocab_size=21,
                          batch_size=50,
                          num_cpu=1,
                          tensorboard_log='nao_logs',
                          input_processing=True,
                          full_tensorboard_log=False
                          )

# === Learn ===

epd_model.learn(X=seqs,
                y=norm_scores,
                epochs=1000,
                eval_interval=50,     # epochs
                log_interval=1,       # epochs
                tb_log_name='seq_search'
                )


# === Save model ===

epd_model.save('nao_seq_search.model')

```
