import gym
import numpy as np
import random

from charles.storage import Storage
from vins.visualize import *

def expert_demos():
    alert('Recording expert demos')

    storage = Storage()

    env = gym.make('ServiceSim-v0')

    s = env.reset()
    for t in range(int(3e4)):
        rate = s[-3]
        instances = s[-2]
        queue = s[-1]

        if queue == 0:
            if random.random() < 0.1:
                a = instances - 1
            else:
                a = instances
        else:
            a = instances + 1

        s2, r, done, _ = env.step(a)

        storage.store((s, np.array([a]), np.array([r]), s2, np.array([done])))

        if t % 100 == 99:
            plot(t, instances, 'Instances', 'Rule-Based')
            plot(t, queue, 'Queue Size', 'Rule-Based')
            plot_reward(t, r, 'Rule-Based')

        s = s2 if not done else env.reset()

    alert('Recording expert demos', done=True)
    return storage
