from vins.expert import expert_demos
from vins.agent import Agent

class ExpertConfig:
    env = 'Hopper-v2'
    actors = 4
    lr = 3e-4
    max_timesteps = 1e5
    trajectory_length = 1
    vis_iter = 500
    storage_size = 1000000
    batch_size = 128
    epochs = 1
    explore_steps = 10000

    testing_steps = 10000

class Config:
    env = 'Hopper-v2'
    vis_iter = 200
    lr = 3e-4
    epochs = 4e4


demos = expert_demos(ExpertConfig)
agent = Agent(Config, demos)
agent.train()



# PARAMS FOR VINS
