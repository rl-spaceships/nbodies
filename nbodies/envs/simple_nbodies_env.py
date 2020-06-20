import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import copy
from .simple_particles import Particle, zero_update_function, gravitational_pull_from_list  # fix to make it work with package, can delete '.' to test in terminal
from numpy import linalg as LA

import logging
logger = logging.getLogger(__name__)

class SimpleNbodiesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, particle_list, step_length, target,
                 max_steps=250, goal_dist=0.1, second_goal_dist=1,
                 goal_reward=5000, large_reward=1000, medium_reward=100, small_reward=10, inverse_dist_reward_power=3,
                 large_penalty=-500, dist_power_penalty=1/15, turn_penalty=-1,
                 bucket_1_max=3, bucket_2_max=10):
        '''

        :param particle_list: list of particles to make environment with first particle being the spaceship
            each particle should be array of inputs to simple_particle.particle
            --> [int id, float mass, float init_time, np.array([], np.float).reshape(2,1) init_position,
            np.array([], np.float).reshape(2,1) init_velocity, float init_fuel, float exhaust_velocity]
            must have all 6 elements and at least 1 particle
        :param step_length: float length of each step of env
        :param target: target for spaceship to reach --> in np.array([], np.float).reshape(2,1)
        :param max_steps: int of number of steps until env is done
        :param goal_dist: float dist to target to have considered reaching goal
        :param second_goal_dist: float dist to target to get some positive reward

        # add infor on reward, penalty params
        # bucket 2 not used
        '''

        self.init_particles = []

        if len(particle_list) == 0:
            raise SyntaxError("particle list can't be empty")
        for i, particle in particle_list:
            if len(particle) != 6:
                raise SyntaxError("The particle {} at index {} in particle list was wrong length".format(particle, i))
            self.init_particles.append(Particle(*particle))

        self.particle_list = copy.deepcopy(self.init_particles)
        self.ship = self.particle_list[0]
        self.target = target

        # create update functions
        self.step_length = step_length
        self.update_functions = [gravitational_pull_from_list(particle_list), zero_update_function]

        # params
        self.action_list = [(0, 0), (1, 0), (1, np.pi), (1, np.pi/2), (1, np.pi * 3 / 2)]
        self.max_steps = max_steps
        self.steps_taken = 0
        self.goal_dist = goal_dist
        self.second_goal_dist = second_goal_dist

        # buckets
        self.bucket_1_max = bucket_1_max
        self.bucket_1_count = 0
        self.bucket_2_max = bucket_2_max
        self.bucket_2_count = 0

        # thetas
        self.thetas = np.arange(0, 2*np.pi, 0.2)

        # goal params
        self.goal_reward, self.large_reward, self.medium_reward = goal_reward, large_reward, medium_reward
        self.small_reward, self.inverse_dist_reward_power = small_reward, inverse_dist_reward_power
        self.large_penalty, self.dist_power_penalty, self.turn_penalty = large_penalty, dist_power_penalty, turn_penalty


        # gym spaces
        self.action_space = spaces.Discrete(5)  # set with {0, ..., 4}
        # 0 - do nothing
        # 1/2 -  +/- x
        # 3/4 -  +/- y

        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=[2,], dtype=np.float64),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=[2,], dtype=np.float64),
            'fuel': spaces.Box(low=0, high=np.inf, shape=[1, ], dtype=np.float64),
            'axis_info': spaces.MultiBinary(2)
        })
        # [x pos, y pos]
        # [x vel, y vel]
        # [fuel]
        # [0/1 pos x, 0/1 pos y]


    # need __init__, step, reset, and render -- might need close

    def step(self, action):
        # action is a number 0-4
        # 0 - do nothing
        # 1/2 -  +/- x
        # 3/4 -  +/- y
        input_thrusts = [(0, 0) for x in range(len(self.particle_list))]
        input_thrusts[0] = self.action_list[action]
        for (particle, update_function, thrust) in \
                zip(self.particle_list, self.update_functions, input_thrusts):
            particle.step(update_function, thrust, self.step_length)
        self.steps_taken += 1
        next_step = self._get_state()
        reward = self._get_reward()
        done = self._is_done()
        log = "TODO do log"

        return next_step, reward, done, log

    def reset(self, theta=None):
        self.steps_taken = 0
        self.particle_list = copy.deepcopy(self.init_particles)
        self.ship = self.particle_list[0]

        # Randomly Initialize Position and Velocity
        old_state = list(self.ship.state_list[-1])
        if theta is None:
            theta = np.random.rand() * 2 * np.pi
            # theta = self.thetas[np.random.randint(len(self.thetas))]
        old_state[1] = np.array([np.sin(theta), np.cos(theta)]).reshape((2, 1))  # position
        old_state[2] = (np.random.rand(2, 1) - .5) * 2  # velocity
        self.ship.state_list[-1] = tuple(old_state)

        # reset buckets
        self.bucket_1_count = 0
        self.bucket_2_count = 0
        return self._get_state()

    def render(self, mode='human', close=False):
        pass

    
    # good to have help functions for other aspects - _take_action, _next_observation, _get_reward

    # def _next_observation(self):
    #     pass
    #
    # def _take_action(self):
    #     pass

    def _get_reward(self):
        distance = self._get_distance_to_goal()
        if distance < self.goal_dist:
            reward = self.goal_reward
        elif self.bucket_1_count < self.bucket_1_max:
            if distance < self.second_goal_dist:
                reward = 1/np.power(distance, self.inverse_dist_reward_power)
                if reward > self.large_reward:
                    reward = self.large_reward
                    self.bucket_1_count += 1
                elif reward > self.medium_reward:
                    reward = self.medium_reward
                else:
                    reward = min(self.small_reward, reward)
            else:
                reward = 1 - np.power(distance, self.dist_power_penalty)
        else:
            reward = self.large_penalty
        return reward - self.turn_penalty

    def _get_state(self):
        current_state = self.ship.current_state()
        return {
            'position': np.array([current_state[0][0], current_state[0][1]]),
            'velocity': np.array([current_state[1][0], current_state[1][1]]),
            'fuel': np.array([current_state[2]]),
            'axis_info': np.array([self._is_positive_axis(1), self._is_positive_axis(0)])
        }

    def _is_positive_axis(self, axis):
        """
        gets whether ship is on a positive axis (x>0 or y>0)

        axis is 0 or 1 where 0 represents x axis and 1 represents y axis

        retrns 1 if on positve axis, 0 if not
        """
        current_state = self.ship.current_state()
        if current_state[axis][0] > self.target[axis]:
            return 1
        else:
            return 0

    def _get_distance_to_goal(self):
        return LA.norm(self.target - self.ship.current_position())

    def _is_done(self):
        return self._get_distance_to_goal() < self.goal_dist or self.max_steps <= self.steps_taken

    def _get_positions(self):
        return [tuple(p.current_position().flatten()) for p in self.particle_list]
