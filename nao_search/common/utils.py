import random

from copy import deepcopy

def random_skills(skill_length, skill_num, action_space=5):
    d = {}
    while len(d) < skill_num:
        skill = [ random.randint(0, action_space) for _ in range(skill_length)]
        d[skill] = 1
    return list(d.keys())


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


def get_top_n(N, skills, scores):
    assert N > 0, ValueError('N must be greater than 0') 
    assert len(skills) > 0, ValueError("The length of the skill list must be greater than 0")
    assert len(skills) == len(scores), ValueError("The skill list and score list have different size")

    skills_bak = deepcopy(skills)
    scores_bak = deepcopy(scores)

    skills_bak, scores_bak = zip(*sorted(zip(scores_bak, skills_bak), reverse=True))

    return list(skills_bak)[:N], list(scores_bak)[:N]


__all__ = [
        random_skills.__name__,
        min_max_normalization.__name__,
        standard_normalization.__name__,
        get_top_n.__name__
        ]
