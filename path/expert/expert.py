import gym
import numpy as np

from charles.storage import Storage
from vins.visualize import alert

def expert_demos():
    alert('Recording expert demos')

    storage = Storage()

    env = gym.make('Path-v0')

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
