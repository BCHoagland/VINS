import torch
import torch.nn as nn
from torch.distributions import Normal

class Network(nn.Module):
    def __init__(self, env, n_obs=None):
        super().__init__()

        self.env = env

        if n_obs is None:
            try:
                self.n_obs = env.observation_space.shape[0]
            except:
                self.n_obs = 1
        else:
            self.n_obs = n_obs

        try:
            self.n_acts = env.action_space.shape[0]
        except:
            self.n_acts = env.action_space.n

        try:
            self.min_a = torch.FloatTensor(env.action_space.low)
            self.max_a = torch.FloatTensor(env.action_space.high)
        except:
            self.min_a = None
            self.max_a = None

class V(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.main = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        s = torch.FloatTensor(s)
        return self.main(s)

class M(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.s = nn.Sequential(
            nn.Linear(self.n_obs, 32),
            nn.ELU(),
        )

        self.a = nn.Sequential(
            nn.Linear(self.n_acts, 32),
            nn.ELU(),
        )

        self.main = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_obs)
        )

    def forward(self, s, a):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)

        s = self.s(s)
        a = self.a(a)

        x = torch.cat((s, a), -1)
        return self.main(x)

class DeterministicPolicy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.mean = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            nn.Tanh()
        )

    def forward(self, s):
        a = self.mean(torch.FloatTensor(s))
        a = ((a + 1) / 2) * (self.max_a - self.min_a) + self.min_a
        return a

class StochasticPolicy(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        self.mean = nn.Sequential(
            nn.Linear(self.n_obs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.n_acts),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(self.n_acts))

    def dist(self, s):
        mean = self.mean(torch.FloatTensor(s))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, s):
        a = self.dist(s).rsample()
        a = ((a + 1) / 2) * (self.max_a - self.min_a) + self.min_a
        return a

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)
