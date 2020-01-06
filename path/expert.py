import torch
import numpy as np
import random

from path.env import PathEnv
from vins.visualize import alert


class Storage:
    def __init__(self):
        self.buffer  = []
    

    def store(self, transition):
        self.buffer.append(transition)
    

    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)
        sampled_transitions = random.sample(self.buffer, batch_size)
        unzipped = list(zip(*sampled_transitions))
        final_samples = [torch.FloatTensor(list(x)) for x in unzipped]
        return final_samples


def expert_demos():
    alert('Recording expert demos')

    storage = Storage()
    env = PathEnv()

    s = env.reset()

    for _ in range(50):
        x, y = s
        if x == 12 and y < 8:
            a = np.array([1])               # up
        else:
            a = np.array([0])               # left

        s2, r, done, _ = env.step(a)

        storage.store((s, a, r, s2, done))

        s = s2 if not done else env.reset()


    alert('Recording expert demos', done=True)
    return storage