import gym
import torch
import pickle
from collections import deque

from storage import Storage
from model import V, M, DeterministicPolicy, StochasticPolicy
from expert import expert_demos
from visualize import *

from charles.models import Model

import sys
sys.path.append('/Users/bradyhoagland/git/gm-sense/environments/service-sim')
import service_sim

lr = 3e-4
vis_iter = 200
epochs = 5e4
demo_steps = 3e4

# move demos to new storage type
with open('.tmp/autoscaler-demos', 'rb') as f:
    storage = pickle.load(f)

def sample(batch_size):
    return [x.view(-1, x.shape[-1]) for x in storage.sample(batch_size)]

# new stuff
env = gym.make('ServiceSim-v0')

v = Model(V, env, lr, target=True)
model = Model(M, env, lr)
π = Model(StochasticPolicy, env, lr)

def fit_value():
    print('Fitting value function and environment model')

    for epoch in range(int(epochs)):
        s, a, r, s2, m = sample(128)

        # calculate value targets
        with torch.no_grad():
            td_target = r + 0.99 * m * v.target(s)

            s_perturb = s + torch.randn_like(s)                                         # may be too much noise
            dist = torch.norm(s - s_perturb, dim=1).unsqueeze(1)
            ns_target = v.target(s) - dist

        # optimize value network
        td_loss = torch.pow(td_target - v(s), 2).mean()
        ns_loss = torch.pow(ns_target - v(s_perturb), 2).mean()
        v_loss = td_loss + ns_loss
        v.optimize(v_loss)

        # optimize model
        model_loss = torch.norm(model(s, a) - s2, dim=1).unsqueeze(1).mean()
        model.optimize(model_loss)

        # update target value network
        v.soft_update_target()

        # plot loss occasionally
        if epoch % vis_iter == vis_iter - 1:
            plot_loss(epoch, v_loss, 'Value', color='#0d0')
            plot_loss(epoch, model_loss, 'Model', color='#0d0')

def clone_behavior():
    print('Fitting behavioral cloning policy')

    for epoch in range(int(epochs)):
        s, a, r, s2, m = sample(128)
        policy_loss = torch.pow(a - π(s), 2).mean()
        π.optimize(policy_loss)

        if epoch % vis_iter == vis_iter - 1:
            plot_loss(epoch, policy_loss, 'π', color='#0d0')

def bc():
    print('Running behavioral cloning policy')

    ep = 0
    all_r = deque(maxlen=1000)
    s = env.reset()
    for t in range(int(demo_steps)):
        with torch.no_grad():
            a = π(s)
            print(a)
        s, r, done, _ = env.step(a.numpy())
        all_r.append(r)

        if t % 500 == 499:
            plot(t, s[-2], 'Instances (BC)')
            plot(t, sum(all_r) / len(all_r), 'BC Mean Reward', color='#fb0')

def noisy_bc():
    print('Running implicit VINS policy')

    all_r = deque(maxlen=1000)
    s = env.reset()
    for t in range(int(demo_steps)):
        with torch.no_grad():
            a = π(s)

            max_a = a
            max_value = v(model(s, max_a))

            for i in range(10):
                noise = 2 * torch.rand_like(a) - 1
                new_a = a + 0.3 * noise
                value = v(model(s, new_a))

                if value > max_value:
                    max_value = value
                    max_a = new_a

        s, r, done, _ = env.step(max_a.numpy())
        all_r.append(r)

        if t % 500 == 499:
            plot(t, s[-2], 'Instances (VINS)')
            plot(t, sum(all_r) / len(all_r), 'VINS Mean Reward', color='#f40')

clone_behavior()
fit_value()
bc()
noisy_bc()
