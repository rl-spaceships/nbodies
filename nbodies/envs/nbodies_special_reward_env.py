import logging
from nbodies.envs.nbodies_env import NbodiesEnv

class NbodiesSpecialRewardEnv(NbodiesEnv):
    
    def __init__(self):
        super(NbodiesEnv, self).__init__()

    def _get_reward(self):
        pass
