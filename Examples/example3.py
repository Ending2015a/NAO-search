import os
import sys
import time
import logging

from nao_search import epd

from nao_search.common.logger import Logger
from nao_search.common.logger import LoggingConfig

from nao_search.common.utils import random_sequences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import get_top_n

LoggingConfig.Use(filename='nao_seq_search.log', output_to_file=True, level='DEBUG')
LOG = Logger()

# === Utils ===
def print_to_file(path, seqs, scores):
    with open(path, 'w') as f:
        for seq, score in zip(seqs, scores):
            f.write(' '.join(map(str, seq)))
            f.write(' {}\n'.format(score))

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

print_to_file('0iter.txt', seqs, scores)

norm_scores = min_max_normalization(scores)

# === Create model ===



epd_model = epd.BaseModel(source_length=60,
                          encoder_vocab_size=21,        # vocab_size + <SOS> = 21
                          decoder_vocab_size=21,        # vocab_size + <SOS> = 21
                          batch_size=50,
                          num_cpu=1,
                          tensorboard_log='nao_logs',
                          input_processing=True,
                          full_tensorboard_log=False
                          )

# === Search ===

Search_epochs = 5

for epoch in range(Search_epochs):

    epd_model.learn(X=seqs,
                    y=norm_scores,
                    epochs=1000,
                    eval_interval=128,     # epochs
                    log_interval=1,        # epochs
                    tb_log_name='seq_search_epoch{:02d}'.format(epoch)
                    )


    epd_model.save('model/nao_seq_search_epoch{:02d}.model'.format(epoch))

    epd_model.eval(X=seqs,
                   y=norm_scores)

    # get top scored seqs
    top_seqs, top_scores = get_top_n(N=100,
                                     seqs=seqs,
                                     scores=scores)

    new_seqs, _ = epd_model.predict(seeds=top_seqs,
                                    lambdas=[10, 20, 30])

    new_scores = []

    for each_seq in new_seqs:
        new_scores.append(get_score(each_seq))

    print_to_file('{}iter.txt'.format(epoch+1), new_seqs, new_scores)

    seqs.extend(new_seqs)
    scores.extend(new_scores)

    norm_scores = min_max_normalization(scores)

    _, top_10_scores = get_top_n(N=10,
                                 seqs=seqs,
                                 scores=scores)
    
    # print top 10 log
    LOG.set_header('Score Ranking')
    LOG.switch_group('Total top 10 ranking')
    for index, score in enumerate(top_10_scores):
        LOG.add_pair('Top {}'.format(index+1), score)

    LOG.switch_group('Top 1 new sequences')
    LOG.add_pair('Score', max(new_scores))

    LOG.dump_to_log()
    

