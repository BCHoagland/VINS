from vins.agent import Agent
import autoscaler
import charles

import sys
sys.path.append('/Users/bradyhoagland/git/gm-sense/environments/service-sim')
import service_sim


class Config:
    env = 'ServiceSim-v0'
    vis_iter = 200
    lr = 3e-4
    epochs = 4e4
    run_steps = 3e4

demos = autoscaler.expert_demos()
agent = Agent(Config, demos)
agent.behavior_clone()
agent.fit_value()
agent.run()
