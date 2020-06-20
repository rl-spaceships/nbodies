from numpy import linalg as LA
import numpy as np


class Particle(object):

    def __init__(self,
                 id,
                 mass,
                 initial_time=0,
                 initial_position=np.zeros((2, 1)),
                 initial_velocity=np.zeros((2, 1)),
                 initial_fuel=0,
                 exhaust_velocity=0):
        if type(id) != int:
            raise TypeError("ID of {} is not an int".format(id))
        if type(mass) != float:
            raise TypeError("mass of {} is not a float".format(mass))
        if type(initial_time) != float:
            raise TypeError("initial_time of {} in not a float".format(initial_time))
        if type(initial_position) != np.array:
            raise TypeError("intial_position of {} is not a np_array".format(initial_position))
        if np.shape(initial_position) != np.zeros((2, 1)).shape:
            raise SyntaxError("the shape of initial_position {} is not (2, 1)".format(np.shape(initial_position)))
        if type(initial_velocity) != np.array:
            raise TypeError("initial_velocity of {} is not a np_array".format(initial_velocity))
        if np.shape(initial_velocity) != np.zeros((2, 1)).shape:
            raise SyntaxError("the shape of initial_velocity {} is not (2, 1)".format(np.shape(initial_velocity)))
        if type(initial_fuel) != float:
            raise TypeError("initial_fuel {} is not a float".format(initial_fuel))
        if initial_fuel < 0:
            raise SyntaxError("initial_fuel {} is less than zero".format(initial_fuel))
        if type(exhaust_velocity) != float:
            raise TypeError("exhaust_velocity {} is not a float".format(exhaust_velocity))
        if exhaust_velocity < 0:
            raise SyntaxError("exhaust_velocity {} is less than zero".format(exhaust_velocity))


        self.id = id  # Particle's UID
        self.mass = mass  # Particle's mass in kilograms
        self.exhaust_velocity = exhaust_velocity


        '''
        List of instantaneous particle states. Each particle state is a tuple
        with the following structure: (time, location, velocity, fuel_weight)
        The dimensions of each of the entries in the tuple are as follows:
        time -> seconds
        location -> n-dimensional numpy vector - meters^2
        velocity -> n-dimensional numpy vector - (meters/second)^2
        fuel_weight -> scalar - kilograms
       '''
        self.state_list = [
            (initial_time, initial_position, initial_velocity, initial_fuel)
        ]

    def current_position(self):
        return self.state_list[-1][1]

    def current_state(self):
        return self.state_list[-1][1:]

    def step(self, update_function, thrust, step_length):
        new_state = update_function(self, thrust, step_length)
        self.state_list.append(new_state)
        # reward = self.get_reward()
        # done = self.is_done()
        # log = "TODO add logging"
        # return new_state, reward, done, log


def zero_update_function(particle, thrust, step_length):
    state = list(particle.state_list[-1])
    state[0] += step_length
    return tuple(state)


def gravitational_pull_from_list(particle_list):
    K = 9.81

    def update_function(particle, thrust, step_length):
        net_f = np.zeros((2, 1))
        time, position, velocity, fuel = particle.state_list[-1]
        m0 = (fuel + particle.mass)
        for p in particle_list:
            if p.id != particle.id:
                distance_p = position - p.current_position()
                force_p = - ((K * m0 * p.mass) / (LA.norm(distance_p)
                                                  ** 2)) * (distance_p / LA.norm(distance_p))
                net_f += force_p

        # Add Thrust Logic Here #
        mag_thrust, dir_thrust = thrust

        if mag_thrust > fuel:
            mag_thrust = fuel
        dv_thrust = particle.exhaust_velocity * np.log(m0 / (m0 - mag_thrust))

        dv_thrust = np.array([dv_thrust * np.cos(dir_thrust), dv_thrust * np.sin(dir_thrust)]).reshape((2, 1))
        #########################
        net_a = net_f / m0
        dv = net_a * step_length
        avg_v = velocity + .5 * dv + dv_thrust
        next_time = time + step_length
        next_position = position + avg_v * step_length
        next_velocity = velocity + dv + dv_thrust
        next_fuel = fuel - mag_thrust

        return (next_time, next_position, next_velocity, next_fuel)

    return update_function