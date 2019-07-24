import random

def random_skills(skill_length, skill_num, action_space=5):
    d = {}
    while len(d) < skill_num:
        skill = [ random.randint(0, action_space) for _ in range(skill_length)]
        d[skill] = 1
    return list(d.keys())


