import random

from copy import deepcopy


def random_sequences(length, seq_num, vocab_size):

    assert vocab_size > 0, ValueError("Vocab_size must be greater than 0")

    d = []
    while len(d) < seq_num:
        seq = [ random.randint(0, vocab_size-1) for _ in range(length)]
        if not seq in d:
            d.append(seq)
    return d


def min_max_normalization(X, lower=0.0, upper=1.0):

    assert len(X) > 0, ValueError("The length of the list X must be greater than 0")

    max_x = max(X)
    min_x = min(X)

    rng = upper - lower
    off = lower

    X_ = [ (x-min_x)/(max_x-min_x) * rng + off for x in X]

    return X_

def standard_normalization(X, lower=0.0, upper=1.0, sigma_clip=2.0):

    assert len(X) > 0, ValueError("The length of the list X must be greater than 0")

    m = sum(X) / len(X)
    var = sum( (x-m)**2 for x in X ) / len(X)

    def clamp(x):
        return sigma_clip if x > sigma_clip else -sigma_clip if x < -sigma_clip else x

    rng = (upper - lower)/(2.0 * sigma_clip) # scaling
    off = (lower + upper)/2.0 # offset

    X_ = [ clamp((x-m)/var) * rng + off for x in X]

    return X_


def get_top_n(N, seqs, scores, reverse=False):
    assert N > 0, ValueError('N must be greater than 0') 
    assert len(seqs) > 0, ValueError("The length of the seq list must be greater than 0")
    assert len(seqs) == len(scores), ValueError("The seq and score list have different size")

    seqs_bak = deepcopy(seqs)
    scores_bak = deepcopy(scores)

    scores_bak, seqs_bak = zip(*sorted(zip(scores_bak, seqs_bak), reverse=not reverse))

    return list(seqs_bak)[:N], list(scores_bak)[:N]


def pairwise_accuracy(la, lb):
    '''
    pairwise_accuract
    https://github.com/renqianluo/NAO/blob/master/NAO/cnn/epd/main.py#L378
    '''
    N = len(la)
    assert N == len(lb)
    total = 0
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            total += 1
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
    return float(count) / total

__all__ = [
        random_sequences.__name__,
        min_max_normalization.__name__,
        standard_normalization.__name__,
        get_top_n.__name__,
        pairwise_accuracy
        ]
