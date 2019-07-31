from vins.model import M
from vins.visualize import *

from charles.models import *
from charles.env import Env

from collections import deque

class Agent:
    def __init__(self, config, demos):
        self.config = config
        self.demos = demos

        self.env = Env(config.env)
        self.reset()


    def reset(self):
        self.v = Model(V, self.env, self.config.lr, target=True)
        self.model = Model(M, self.env, self.config.lr)
        self.π_bc = Model(LinearPolicy, self.env, self.config.lr)


    def train(self):
        self.env = Env(self.config.env, self.config.actors)
        for env_wrapper in self.algo.env_wrappers:
            self.env = env_wrapper(self.env)
        super().train(setup=False)


    def behavior_clone(self, epochs=None):
        alert('Behavioral cloning')

        if epochs is None: epochs = self.config.epochs

        for epoch in range(int(epochs)):
            s, a, r, s2, m = self.demos.sample(128)
            if self.env.action_space.__class__.__name__ == 'Discrete':
                policy_loss = -self.π_bc.log_prob(s, a).mean()
            else:
                policy_loss = torch.pow(a - self.π_bc(s), 2).mean()
            self.π_bc.optimize(policy_loss)

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
                    s_perturb = s + 1 * torch.randn_like(s)                                                         # may be too much noise
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
        a = self.π_bc(s)
        if self.env.action_space.__class__.__name__ == 'Discrete': a = a.unsqueeze(1)

        max_a = a
        max_value = self.v(self.model(s, max_a))

        for i in range(10):
            # noise = 2 * torch.rand_like(a) - 1
            # new_a = a + 15 * noise
            new_a = torch.FloatTensor(self.env.action_space.sample()).unsqueeze(1)

            value = self.v(self.model(s, new_a))

            if value > max_value:
                max_value = value
                max_a = new_a
        return max_a


    def demo(self, policy='VINS', mean_reward=False):
        alert(f'Running {policy} policy')

        if policy is 'VINS':
            get_action = self.vins_action
            color = '#5106E099'
        elif policy is 'BC':
            get_action = self.π_bc
            color = '#B466FF99'
        else:
            print('That policy doesn\'t exist :(')
            quit()

        ep_r = 0
        final_ep_r = 0
        past_r = deque(maxlen=1000)
        s = self.env.reset()

        for t in range(int(self.config.run_steps)):
            with torch.no_grad():
                a = get_action(s)

            s, r, done, _ = self.env.step(a)
            ep_r += r
            past_r.append(r)

            if t % self.config.vis_iter == self.config.vis_iter - 1:
                if mean_reward:
                    plot_reward(t, sum(past_r) / len(past_r), policy, color=color)
                else:
                    plot_reward(t, final_ep_r, policy, color=color)

                instances = s[0][-2]
                queue = s[0][-1]
                plot(t, instances, 'Instances', policy, color=color)
                plot(t, queue, 'Queue Size', policy, color=color)

                if policy is 'VINS':
                    with torch.no_grad():
                        plot(t, self.v(self.model(s, a)), 'Predicted Return', policy, color=color)

            if done:
                final_ep_r = ep_r
                ep_r = 0
                s = self.env.reset()

        alert(f'Running {policy} policy', done=True)


    def run(self, mean_reward=False):
        self.demo('BC', mean_reward)
        self.demo('VINS', mean_reward)


    def map_value(self, name):
        map(self.v, self.env.observation_space.low, self.env.observation_space.high, name)
