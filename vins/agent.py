from vins.models import Network, ValueNetwork, EnvironmentModel, StochasticPolicy
from vins.visualize import *
from path import PathEnv

from collections import deque
import numpy as np

class Agent:
    def __init__(self, config, demos):
        self.config = config
        self.demos = demos

        self.env = PathEnv()
        self.reset()


    def reset(self):
        self.v = Network(ValueNetwork, self.config.lr, target=True)
        self.model = Network(EnvironmentModel, self.config.lr)
        self.π_bc = Network(StochasticPolicy, self.config.lr)


    def behavior_clone(self, epochs):
        alert('Behavioral cloning')

        for epoch in range(int(epochs)):
            s, a, r, s2, d = self.demos.sample(128)
            m = 1 - d
            policy_loss = -self.π_bc.log_prob(s, a).mean()                          # for discrete action spaces only
            self.π_bc.minimize(policy_loss)

            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, policy_loss, 'π', '#B466FF')

        alert('Behavioral cloning', done=True)


    def fit_value(self, epochs, negative_sampling=True):
        alert('Fitting value function')

        name_mod = 'NS' if negative_sampling else 'Normal'

        for epoch in range(int(epochs)):
            s, a, r, s2, d = self.demos.sample(128)
            m = 1 - d

            # calculate value targets
            with torch.no_grad():
                td_target = r + 0.99 * m * self.v.target(s2)
            v_loss = torch.pow(td_target - self.v(s), 2).mean()

            if negative_sampling:
                with torch.no_grad():
                    s_perturb = s + 1 * torch.randn_like(s)                                                         # may be too much noise
                    dist = torch.norm(s - s_perturb, dim=1).unsqueeze(1)
                    ns_target = self.v.target(s) - dist
                v_loss += torch.pow(ns_target - self.v(s_perturb), 2).mean()

            # optimize value network
            self.v.minimize(v_loss)

            # optimize model
            model_loss = torch.norm(self.model(s, a) - s2, dim=1).unsqueeze(1).mean()
            self.model.minimize(model_loss)

            # update target value network
            self.v.soft_update_target()

            # plot loss occasionally
            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, v_loss, f'Value - {name_mod}', '#5106E0')
                plot_loss(epoch, model_loss, f'Model - {name_mod}', '#5106E0')

        alert('Fitting value function', done=True)


    def vins_action(self, s):
        a = self.π_bc(s).unsqueeze(-1)

        max_a = a
        max_value = self.v(self.model(s, max_a))

        for i in range(10):
            # noise = 2 * torch.rand_like(a) - 1
            # new_a = a + 15 * noise
            new_a = torch.FloatTensor([self.env.random_action()])

            value = self.v(self.model(s, new_a))

            if value > max_value:
                max_value = value
                max_a = new_a
        return max_a


    def demo(self, steps, policy='VINS', mean_reward=False):
        alert(f'Running {policy} policy')

        if policy is 'VINS':
            get_action = self.vins_action
            color = '#5106E099'
        elif policy is 'BC':
            get_action = self.π_bc
            color = '#B466FF99'
        else:
            print('That policy doesn\'t exist')
            quit()

        ep_r = 0
        final_ep_r = 0
        past_r = deque(maxlen=1000)
        s = self.env.reset()

        for t in range(int(steps)):
            with torch.no_grad():
                a = get_action(torch.FloatTensor(s))

            s, r, done, _ = self.env.step(a)
            ep_r += r
            past_r.append(r)

            if t % self.config.vis_iter == self.config.vis_iter - 1:
                if mean_reward:
                    plot_reward(t, sum(past_r) / len(past_r), policy, color=color)
                else:
                    plot_reward(t, final_ep_r, policy, color=color)

                if policy is 'VINS':
                    with torch.no_grad():
                        plot(t, self.v(self.model(torch.FloatTensor(s), a)), 'Predicted Return', policy, color=color)

            if done:
                final_ep_r = ep_r
                ep_r = 0
                s = self.env.reset()

        alert(f'Running {policy} policy', done=True)


    def run(self, steps, mean_reward=False):
        self.demo(steps, 'BC', mean_reward)
        self.demo(steps, 'VINS', mean_reward)


    def map_value(self, name):                                                          #! move somewhere else (PathEnv maybe?)
        map(self.v, np.array([0, 0]), np.array([13, 11]), name)
