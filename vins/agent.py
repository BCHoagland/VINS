import gym

from vins.model import V, M, DeterministicPolicy, StochasticPolicy
from vins.visualize import *

from charles.models import Model

class Agent:
    def __init__(self, config, demos):
        self.config = config
        self.demos = demos

        self.env = gym.make(self.config.env)

        self.v = Model(V, self.env, self.config.lr, target=True)
        self.model = Model(M, self.env, self.config.lr)
        self.π = Model(StochasticPolicy, self.env, self.config.lr)

    def behavior_clone(self):
        alert('Behavioral cloning')

        for epoch in range(int(self.config.epochs)):
            s, a, r, s2, m = self.demos.sample(128)
            policy_loss = torch.pow(a - self.π(s), 2).mean()
            self.π.optimize(policy_loss)

            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, policy_loss, 'π', '#B466FF')

        alert('Behavioral cloning', done=True)

    def fit_value(self):
        alert('Running conservative extrapolation')

        for epoch in range(int(self.config.epochs)):
            s, a, r, s2, m = self.demos.sample(128)

            # calculate value targets
            with torch.no_grad():
                td_target = r + 0.99 * m * self.v.target(s)

                s_perturb = s + torch.randn_like(s)                                                         # may be too much noise
                dist = torch.norm(s - s_perturb, dim=1).unsqueeze(1)
                ns_target = self.v.target(s) - dist

            # optimize value network
            td_loss = torch.pow(td_target - self.v(s), 2).mean()
            ns_loss = torch.pow(ns_target - self.v(s_perturb), 2).mean()
            v_loss = td_loss + ns_loss
            self.v.optimize(v_loss)

            # optimize model
            model_loss = torch.norm(self.model(s, a) - s2, dim=1).unsqueeze(1).mean()
            self.model.optimize(model_loss)

            # update target value network
            self.v.soft_update_target()

            # plot loss occasionally
            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, v_loss, 'Value', '#5106E0')
                plot_loss(epoch, model_loss, 'Model', '#5106E0')

        alert('Running conservative extrapolation', done=True)

    def vins_action(self, s):
        a = self.π(s)

        max_a = a
        max_value = self.v(self.model(s, max_a))

        for i in range(10):
            noise = 2 * torch.rand_like(a) - 1
            new_a = a + 0.3 * noise
            value = self.v(self.model(s, new_a))

            if value > max_value:
                max_value = value
                max_a = new_a

        return max_a

    def demo(self, policy='VINS'):
        alert(f'Running {policy} policy')

        if policy is 'VINS':
            get_action = self.vins_action
            color = '#5106E0'
        else:
            get_action = self.π
            color = '#B466FF'

        ep = 0
        ep_r = 0
        s = self.env.reset()

        for _ in range(int(1e4)):
            with torch.no_grad():
                a = get_action(s)

            s, r, done, _ = self.env.step(a.numpy())
            ep_r += r

            if done:
                ep += 1
                plot_reward(ep, ep_r, policy, color=color)
                ep_r = 0
                s = self.env.reset()

        alert(f'Running {policy} policy', done=True)

    def train(self):
        self.behavior_clone()
        self.fit_value()

        self.demo('BC')
        self.demo('VINS')
