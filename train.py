import gym
import torch

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
epochs = 2e4

# move demos to new storage type
demos = expert_demos()
storage = Storage()
storage.buffer = demos.buffer                                                           # do I need this anymore?

def sample(batch_size):
    return [x.view(-1, x.shape[-1]) for x in storage.sample(batch_size // 4)]

# new stuff
env = gym.make('Hopper-v2')

v = Model(V, env, lr, target=True)
model = Model(M, env, lr)
μ = Model(DeterministicPolicy, env, lr)
π = Model(StochasticPolicy, env, lr)

def fit_value():
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

        # optimize policy
        policy_loss = -v(model(s, μ(s))).mean()
        μ.optimize(policy_loss)

        # update target value network
        v.soft_update_target()

        # plot loss occasionally
        if epoch % vis_iter == vis_iter - 1:
            plot_loss(epoch, v_loss, 'Value')
            plot_loss(epoch, model_loss, 'Model')

def clone_behavior():
    for epoch in range(int(epochs)):
        s, a, r, s2, m = sample(128)
        policy_loss = torch.pow(a - π(s), 2).mean()
        π.optimize(policy_loss)

        if epoch % vis_iter == vis_iter - 1:
            plot_loss(epoch, policy_loss, 'π')

def noisy_bc():
    ep = 0
    ep_r = 0
    s = env.reset()
    for _ in range(int(1e4)):
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
        ep_r += r
        if done:
            ep += 1
            plot_reward(ep, ep_r, 'VINS')
            ep_r = 0
            s = env.reset()

def demo():
    ep = 0
    ep_r = 0
    s = env.reset()
    for _ in range(int(1e4)):
        with torch.no_grad():
            # a = μ(s)
            a = π(s)
        s, r, done, _ = env.step(a.numpy())
        ep_r += r
        if done:
            ep += 1
            plot_reward(ep, ep_r, 'BC')
            ep_r = 0
            s = env.reset()

clone_behavior()
demo()
fit_value()
noisy_bc()
