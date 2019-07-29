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


def min_max_normalization(X):

    assert len(X) > 0, ValueError("The length of the list X must be greater than 0")

    max_x = max(X)
    min_x = min(X)

    X_ = [ (x-min_x)/(max_x-min_x) for x in X]

    return X_

def standard_normalization(X):

    assert len(X) > 0, ValueError("The length of the list X must be greater than 0")

    m = sum(X) / len(X)
    var = sum( (x-m)**2 for x in X ) / len(X)

    def clamp(x):
        return 2.0 if x > 2.0 else -2.0 if x < -2.0 else x

    X_ = [ clamp((x-m)/var)*0.25+0.5 for x in X]

    return X_


def get_top_n(N, seqs, scores, reverse=False):
    assert N > 0, ValueError('N must be greater than 0') 
    assert len(seqs) > 0, ValueError("The length of the seq list must be greater than 0")
    assert len(seqs) == len(scores), ValueError("The seq and score list have different size")

    seqs_bak = deepcopy(seqs)
    scores_bak = deepcopy(scores)

    scores_bak, seqs_bak = zip(*sorted(zip(scores_bak, seqs_bak), reverse=not reverse))

    return list(seqs_bak)[:N], list(scores_bak)[:N]


__all__ = [
        random_sequences.__name__,
        min_max_normalization.__name__,
        standard_normalization.__name__,
        get_top_n.__name__
        ]
