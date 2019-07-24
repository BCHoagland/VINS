import torch
import torch.nn as nn

from charles.models import Network

class M(Network):
    def __init__(self, env, n_obs=None):
        super().__init__(env, n_obs)

        if env.action_space.__class__.__name__ == 'Discrete':
            a_in = 1
        else:
            a_in = self.n_acts

        self.s = nn.Sequential(
            nn.Linear(self.n_obs, 32),
            nn.ELU(),
        )

        self.a = nn.Sequential(
            nn.Linear(a_in, 32),
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
