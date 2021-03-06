from vins.models import Network, ValueNetwork, EnvironmentModel, StochasticPolicy
from vins.visualize import *
from path import PathEnv

from collections import deque
import numpy as np




######################
# Code Abbreviations #
######################
# s : state
# s2 : next state
# a : action
# r : reward
# d : done
# m : mask (opposite of done)

# v : value (representing a value network)
# π_bc : stochastic policy found through behavioral cloning

# ns : negative sampling

# ep : episode




class Agent:
    def __init__(self, config, demos):
        self.config = config
        self.demos = demos

        self.env = PathEnv()
        self.reset()


    # create new networks for the agent to use
    # see vins/models.py for implementation details
    def reset(self):
        self.v = Network(ValueNetwork, self.config.lr, target=True)
        self.model = Network(EnvironmentModel, self.config.lr)
        self.π_bc = Network(StochasticPolicy, self.config.lr)


    # use expectation maximization to learn a stochastic policy from expert demonstrations
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


    # fit a value function from expert demonstrations, with an option to use negative sampling
    def fit_value(self, epochs, negative_sampling=True):
        alert_mod = '(negative sampling)' if negative_sampling else '(normal)'
        alert('Fitting value function ' + alert_mod)

        name_mod = 'NS' if negative_sampling else 'Normal'

        for epoch in range(int(epochs)):
            s, a, r, s2, d = self.demos.sample(128)
            m = 1 - d

            # calculate value targets
            with torch.no_grad():
                td_target = r + 0.99 * m * self.v.target(s2)
            v_loss = torch.pow(td_target - self.v(s), 2).mean()

            # negative sampling to augment value loss
            if negative_sampling:
                with torch.no_grad():
                    s_perturb = s + 1 * torch.randn_like(s)                                                         # may be too much noise
                    dist = torch.norm(s - s_perturb, dim=1).unsqueeze(1)
                    ns_target = self.v.target(s) - dist
                v_loss += torch.pow(ns_target - self.v(s_perturb), 2).mean()

            # optimize value network
            self.v.minimize(v_loss)

            # optimize environment model
            model_loss = torch.norm(self.model(s, a) - s2, dim=1).unsqueeze(1).mean()
            self.model.minimize(model_loss)

            # update target value network
            self.v.soft_update_target()

            # plot loss occasionally
            if epoch % self.config.vis_iter == self.config.vis_iter - 1:
                plot_loss(epoch, v_loss, f'Value - {name_mod}', '#5106E0')
                plot_loss(epoch, model_loss, f'Model - {name_mod}', '#5106E0')

        alert('Fitting value function ' + alert_mod, done=True)


    # use random shooting based on the value function to determine an action
    def vins_action(self, s):
        a = self.π_bc(s).unsqueeze(-1)

        max_a = a
        max_value = self.v(self.model(s, max_a))

        for i in range(10):
            # the paper adds random noise to the BC policy
            # the path environment is discrete, though, so I sample actions instead
            new_a = torch.FloatTensor([self.env.random_action()])

            value = self.v(self.model(s, new_a))

            if value > max_value:
                max_value = value
                max_a = new_a
        return max_a


    # get a running reward plot on the environment using a given policy
    def demo(self, steps, policy='VINS'):
        alert(f'Running {policy} policy')

        # decide how to choose actions
        if policy is 'VINS':
            get_action = self.vins_action
            color = '#5106E099'
        elif policy is 'BC':
            get_action = self.π_bc
            color = '#B466FF99'
        else:
            print('That policy doesn\'t exist')
            quit()
        
        
        # run through the environment
        successes = 0
        s = self.env.reset()
        for t in range(int(steps)):
            with torch.no_grad():
                a = get_action(torch.FloatTensor(s))

            s, _, done, _ = self.env.step(a)

            # occassionally plot how many times we've reached the target
            if t % self.config.vis_iter == self.config.vis_iter - 1:
                plot(t, successes, 'Successes', policy, color=color)

            # if we're done, then we reached the target successfully
            if done:
                successes += 1
                s = self.env.reset()

        alert(f'Running {policy} policy', done=True)


    # run both BC and VINS policies and plot the rewards
    def run(self, steps):
        self.demo(steps, 'BC')
        self.demo(steps, 'VINS')

    # plot a map of the value function
    def map_value(self, name):
        map(self.v, np.array([0, 0]), np.array([13, 11]), name)
