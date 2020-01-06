from vins.agent import Agent
import path

class Config:
    vis_iter = 200
    lr = 3e-4


# make VINS agent using expert demos
demos = path.expert_demos()
agent = Agent(Config, demos)

# run value function extrapolation using standard TD error
agent.fit_value(epochs=1e4, negative_sampling=False)
agent.map_value('Normal')

# reset the agent and run conservative value function extrapolation
agent.reset()
agent.fit_value(epochs=4e4)
agent.map_value('NS')

# run behavioral cloning to get base policy
agent.behavior_clone(epochs=1e4)

# demo the BC policy and the implicit VINS policy
agent.run(1e3)