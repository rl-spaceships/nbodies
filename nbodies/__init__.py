import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='nbodies-simple-v0',
    entry_point='nbodies.envs:SimpleNbodiesEnv',
)

register(
    id='nbodies-v0',
    entry_point='nbodies.envs:NbodiesEnv',
)

register(
    id='nbodies-specialReward-v0'
    entry_point='nbodies.envs:NbodiesSpecialRewardEnv'
)

# example of how can add another type of env
#register(
#    id='SoccerEmptyGoal-v0',
#    entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
#    timestep_limit=1000,
#    reward_threshold=10.0,
#    nondeterministic = True,
#)
