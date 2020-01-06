import numpy as np
import random
from math import floor

class PathEnv():

    def __init__(self):
        self.directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self._init_state()

    def _init_state(self):
        self.x = 12
        self.y = 1

    def _obs(self):
        return np.array([self.x, self.y])

    def _reward(self):
        r = 0 if all(self._done()) else -1
        return np.array([r])

    def _done(self):
        return np.array([self.x == 3 and self.y == 8])

    def step(self, a):
        '''
        0: LEFT
        1: UP
        2: RIGHT
        3: DOWN
        '''

        try:
            a = a[0]
        except:
            pass

        dir = self.directions[int(a)]

        self.x += dir[0]
        self.y += dir[1]

        return self._obs(), self._reward(), self._done(), {}

    def reset(self):
        self._init_state()
        return self._obs()

    def random_action(self):
        return int(random.random() * 4)