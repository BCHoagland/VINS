from vins.model import M
from vins.visualize import *

from charles.models import *
from charles.env import Env

class Agent:
    def __init__(self, config, demos):
        self.config = config
        self.demos = demos

        self.env = Env(self.config.env)
        self.reset()

    def behavior_clone(self, epochs=None):
        alert('Behavioral cloning')

        if epochs is None: epochs = self.config.epochs

        for epoch in range(int(epochs)):
            s, a, r, s2, m = self.demos.sample(128)
            # policy_loss = torch.pow(a - self.π(s), 2).mean()
            policy_loss = -self.π.log_prob(s, a).mean()
            self.π.optimize(policy_loss)

            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, policy_loss, 'π', '#B466FF')

        alert('Behavioral cloning', done=True)

    def fit_value(self, epochs=None, negative_sampling=True):
        alert('Fitting value function')

        if epochs is None: epochs = self.config.epochs
        name_mod = 'NS' if negative_sampling else 'Normal'

        for epoch in range(int(epochs)):
            s, a, r, s2, m = self.demos.sample(128)

            # calculate value targets
            with torch.no_grad():
                td_target = r + 0.99 * m * self.v.target(s2)
            v_loss = torch.pow(td_target - self.v(s), 2).mean()

            if negative_sampling:
                with torch.no_grad():
                    s_perturb = s + 3 * torch.randn_like(s)                                                         # may be too much noise
                    dist = torch.norm(s - s_perturb, dim=1).unsqueeze(1)
                    ns_target = self.v.target(s) - dist
                v_loss += torch.pow(ns_target - self.v(s_perturb), 2).mean()

            # optimize value network
            self.v.optimize(v_loss)

            # optimize model
            model_loss = torch.norm(self.model(s, a) - s2, dim=1).unsqueeze(1).mean()
            self.model.optimize(model_loss)

            # update target value network
            self.v.soft_update_target()

            # plot loss occasionally
            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, v_loss, f'Value - {name_mod}', '#5106E0')
                plot_loss(epoch, model_loss, f'Model - {name_mod}', '#5106E0')

        alert('Fitting value function', done=True)

    def vins_action(self, s):
        a = self.π(s)
        if self.env.action_space.__class__.__name__ == 'Discrete': a = a.unsqueeze(1)

        max_a = a
        max_value = self.v(self.model(s, max_a))

        for i in range(10):
            noise = 2 * torch.rand_like(a) - 1
            new_a = a + 2 * noise
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

        for _ in range(int(self.config.run_steps)):
            with torch.no_grad():
                a = get_action(s)

            s, r, done, _ = self.env.step(a)
            ep_r += r

            if done:
                ep += 1
                plot_reward(ep, ep_r, policy, color=color)
                ep_r = 0
                s = self.env.reset()

        alert(f'Running {policy} policy', done=True)

    def reset(self):
        self.v = Model(V, self.env, self.config.lr, target=True)
        self.model = Model(M, self.env, self.config.lr)
        self.π = Model(LinearPolicy, self.env, self.config.lr)

    def run(self):
        self.demo('BC')
        self.demo('VINS')

    def map_value(self, name):
        map(self.v, self.env.observation_space.low, self.env.observation_space.high, name)
