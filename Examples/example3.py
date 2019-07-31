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

# === params ===

Search_epochs = 5
Learning_epochs = 1000
Eval_interval = 100
Log_interval = 1

Top_n = 50
Lambdas = [10, 20, 30, 40, 50, 60]

Log_filename = 'nao_seq_search_clip37'
Tensorboard_logdir = 'nao_logs'
Tensorboard_logname = 'nao_seq_search_clip37'
Searched_seqs_output_path = 'output/{}'.format(Log_filename)
Model_save_path = 'model/{}'.format(Log_filename)


Norm_clip_min = 0.3
Norm_clip_max = 0.7

# ==============

LoggingConfig.Use(filename='{}.log'.format(Log_filename), output_to_file=True, level='DEBUG')
LOG = Logger()

# === Utils ===

def get_score(seq):
    return sum(seq)

def make_dirs(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def print_to_file(path, seqs):
    dirname = os.path.dirname(path)

    make_dirs(dirname)

    with open(path, 'w') as f:
        for seq, score in zip(seqs, scores):
            f.write(' '.join(map(str, seq)))
            f.write('\n')
# === Generate sequences ===

seqs = random_sequences(length=60,
                        seq_num=300,
                        vocab_size=20)


# === Calculate scores ===

scores = []
for seq in seqs:
    scores.append(get_score(seq))

print_to_file(os.path.join(Searched_seqs_output_path, '0iter.txt'), seqs)

norm_scores = min_max_normalization(scores, Norm_clip_min, Norm_clip_max)

# === Create model ===



epd_model = epd.BaseModel(source_length=60,
                          encoder_vocab_size=21,        # vocab_size + <SOS> = 21
                          decoder_vocab_size=21,        # vocab_size + <SOS> = 21
                          batch_size=100,
                          num_cpu=1,
                          tensorboard_log=Tensorboard_logdir,
                          input_processing=True,
                          full_tensorboard_log=False
                          )

# === Search ===

for epoch in range(Search_epochs):

    epd_model.learn(X=seqs,
                    y=norm_scores,
                    epochs=Learning_epochs,
                    eval_interval=Eval_interval,      # epochs
                    log_interval=Log_interval,        # epochs
                    tb_log_name='{}_epoch{:02d}'.format(Tensorboard_logname, epoch+1)
                    )


    epd_model.save('{}_epoch{:02d}.model'.format(Model_save_path, epoch+1))

    epd_model.eval(X=seqs,
                   y=norm_scores)

    # get top scored seqs
    top_seqs, top_scores = get_top_n(N=Top_n,
                                     seqs=seqs,
                                     scores=scores)

    new_seqs, _ = epd_model.predict(seeds=top_seqs,
                                    lambdas=Lambdas)

    new_scores = []

    for each_seq in new_seqs:
        new_scores.append(get_score(each_seq))

    print_to_file(os.path.join(Searched_seqs_output_path, '{}iter.txt'.format(epoch+1)), new_seqs)



    # === print top 10 log ===

    LOG.set_header('Epoch {}/{} Score Ranking'.format(epoch+1, Search_epochs))

    # === print top 10 for each lambda

    for index, lamb in enumerate(Lambdas):
        start_ind = index * Top_n
        end_ind = start_ind + Top_n

        LOG.switch_group('Top 10 scores for lambda={}'.format(lamb))

        _, top_10_scores = get_top_n(N=10,
                                     seqs=new_seqs[start_ind:end_ind],
                                     scores=new_scores[start_ind:end_ind])

        for ind, score in enumerate(top_10_scores):
            LOG.add_pair('Top {}'.format(ind+1), score)

    # === print top 10 out of newly generated seqs
    
    LOG.switch_group('Top 10 ranking (New seqs)')

    _, top_10_scores = get_top_n(N=10,
                                 seqs=new_seqs,
                                 scores=new_scores)

    for index, score in enumerate(top_10_scores):
        LOG.add_pair('Top {}'.format(index+1), score)

    # === print top 10 out of old seqs

    LOG.switch_group('Old top 10 ranking (Summary)')

    _, top_10_scores = get_top_n(N=10,
                                 seqs=seqs,
                                 scores=scores)

    for index, score in enumerate(top_10_scores):
        LOG.add_pair('Top {}'.format(index+1), score)


    # add newly generated seqs
    seqs.extend(new_seqs)
    scores.extend(new_scores)

    # normalize
    norm_scores = min_max_normalization(scores, Norm_clip_min, Norm_clip_max)


    # === print top 10 out of all seqs (old+new)

    LOG.switch_group('Top 10 ranking (Summary)')

    _, top_10_scores = get_top_n(N=10,
                                 seqs=seqs,
                                 scores=scores)

    for index, score in enumerate(top_10_scores):
        LOG.add_pair('Top {}'.format(index+1), score)

    LOG.dump_to_log()
    

