# NAO Skill Search

## Introduction
A simplified tensorflow version of Neural Architecture Optimization.

## Example
```python
import os
import sys
import time
import logging

from nao_search import epd
from nao_search.common import LoggingConfig
from nao_search.common.utils import random_skills, min_max_normalization, get_top_n

ACTION_SPACE = 5
SKILL_LENGTH = 5
SKILL_NUM = 500
TOP_N = 100
LAMBDAS = [10, 20, 30]
SEARCH_EPOCHS = 5
LEARING_EPOCHS = 10

# generate random skills
skills = random_skills(
            skill_length=SKILL_LENGTH, 
            skill_num=SKILL_NUM,
            action_space=ACTION_SPACE)


scores = []
for each_skill in skills:
    #TODO: evaluate your skill's score
    scores.append(evaluate_your_skill(each_skill))

# normalize your scores between 0.0 ~ 1.0
norm_scores = min_max_normalization(scores)

# create epd model
epd_model = epd.Model(skill_length=SKILL_LENGTH)

for epoch in range(SEARCH_EPOCHS):
    # start training
    epd_model.learn(skills, norm_scores, epochs=LEARNING_EPOCHS)
    
    # get top N scored skills
    seeds, _scores = get_top_n(
                            N=TOP_N,
                            skills=skills,
                            scores=scores)
    
    # predict new skills
    new_skills = epd_model.predict(seeds=seeds, lambdas=LAMBDAS)
    
    for each_new_skill in new_skills:
        skills.append(each_new_skill)
        # TODO: evaluate your skill's score
        scores.append(evaluate_your_score(each_new_skill))
        
    norm_scores = min_max_normalization(scores)
    
top_10_skills, _scores = get_top_n(
                                N=10,
                                skills=skills,
                                scores=scores)
                        
for index, (skill, score) in enumerate(zip(top_10_skills, _scores)):
    print('Top {} skill: {}, score: {}'.format(index+1, skill, score))
```
