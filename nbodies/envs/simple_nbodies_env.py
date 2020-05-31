import gym
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

class SimpleNbodiesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass
        # define self.action_space and self.observation_space
        # must use gym.spaces objects

    # need __init__, step, reset, and render -- might need close

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    
    # good to have help functions for other aspects - _take_action, _next_observation, _get_reward

    _next_observation(self):
        pass

    _take_action(self):
        pass

    _get_reward(self):
        pass
