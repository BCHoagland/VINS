from vins.expert import expert_demos
from vins.agent import Agent

import path

class Config:
    env = 'Path-v0'
    vis_iter = 200
    lr = 3e-4
    epochs = 4e4
    run_steps = 1e3


demos = path.expert_demos()
agent = Agent(Config, demos)

agent.fit_value(epochs=1e4, negative_sampling=False)
agent.map_value('Normal')

agent.reset()
agent.fit_value()
agent.map_value('NS')

agent.behavior_clone(epochs=1e4)

agent.run()

# PARAMS FOR VINS
