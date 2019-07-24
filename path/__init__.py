from path.expert.expert import expert_demos
from gym.envs.registration import register

register(
    id='Path-v0',
    entry_point='path.envs:PathEnv'
)
