import os
import sys
import time
import logging

from nao_search import epd
from nao_search.common import LoggingConfig #done
from nao_search.common.utils import random_skills #done

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy

from stable_baselines.common.vec_env import SubprocVecEnv

# === your game ===
from nes.py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros as gym_mario

from gym_mario.actions import COMPLEX_MOVEMENT
from skill_wrapper import RewardWrapper

def make_env():
    env = gym_mario.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    env = RewardWrapper(env, gamma=0.9)
    return env
# =================

# ===== Hyperparameters =====


# ===========================

# ===== Other Parameters ====



# ===========================


LoggingConfig.Use(filename='searching.log', level='INFO')
LOG = logging.getLogger()

def makedirs(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


'''
How to use epd model

* Create new epd model

    epd_model = epd.Model()

* Load epd model from file

    epd_model = epd.load('my_epd.model')
    
    or

    epd_model = epd.Model()
    epd_model.load('my_epd.model')

* Save your model

    epd_model.save('my_new_epd.model')

* Train your epd model

    skills = random_skills(...)
    scores = ...

    epd_model.fit(skills, scores)

* Predict new skills
    
    seed_skills = getTop50(skills, scores)
    epd_model.predict(seeds=seed_skills, lambdas=[10, 20, 30])

* Predict score
    
    score = epd_model.predict_score(skill)

'''
