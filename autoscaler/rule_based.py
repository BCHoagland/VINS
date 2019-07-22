import sys
sys.path.append('/Users/bradyhoagland/git/gm-sense/environments/service-sim')
import service_sim

import gym
import numpy as np
import random
import pickle

from storage import Storage
from visualize import *


timesteps = 50000
vis_iter = 100


env = gym.make('ServiceSim-v0')

storage = Storage()

s = env.reset()
for t in range(int(timesteps)):

    # rule-based action selection
    req_rate = s[-3]
    instances = s[-2]
    requests = s[-1]
    if requests == 0:
        if random.random() < 0.1:
            a = instances - 1
        else:
            a = instances
    else:
        a = instances + 1

    s2, r, done, _ = env.step(a)

    storage.store((s, np.array([a]), np.array([r]), s2, np.array([done])))

    s = s2 if not done else env.reset()

    if t % vis_iter == vis_iter - 1:
        plot(t, req_rate, 'Rule-Based Request Rate', '#f0f')
        plot(t, instances, 'Rule-Based Instances', '#f00')
        plot(t, requests, 'Rule-Based Requests', '#00f')

with open('.tmp/autoscaler-demos', 'wb') as f:
    pickle.dump(storage, f)
