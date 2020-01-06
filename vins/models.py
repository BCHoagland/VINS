import torch
import torch.nn as nn
from torch.distributions import Categorical


class Network:
    def __init__(self, network_type, lr, target=False):
        self.net = network_type()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        if target:
            self.target_net = network_type()
            self.target_net.load_state_dict(self.net.state_dict())
    
    def __call__(self, *args):
        return self.net(*args)

    def minimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def maximize(self, loss):
        self.optim.zero_grad()
        (-loss).backward()
        self.optim.step()

    def target(self, *args):
        return self.target_net(*args)

    def soft_update_target(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_((0.995 * target_param.data) + ((0.005) * param.data))
    
    def log_prob(self, s, a):
        return self.net.log_prob(s, a)



#! add note about why states are 2-dimensional, actions are 4-dim...





class EnvironmentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # state preprocessing
        self.s = nn.Sequential(
            nn.Linear(2, 32),
            nn.ELU(),
        )

        # action preprocessing
        self.a = nn.Sequential(
            nn.Linear(1, 32),
            nn.ELU(),
        )

        # next-state predictor
        self.main = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2)
        )

    def forward(self, s, a):
        s = self.s(s)
        a = self.a(a)

        # print('---------')
        # print(s.shape)
        # print(a.shape)
        # print('---------')
        x = torch.cat((s, a), -1)
        return self.main(x)


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    

    def forward(self, s):
        return self.main(s)


class StochasticPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.logits = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def dist(self, s):
        return Categorical(logits=self.logits(s))
    
    def forward(self, s):
        dist = self.dist(s)
        return self.dist(s).sample().float()
    
    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)