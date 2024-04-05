import numpy as np
import matplotlib.pyplot as plt
import logging as logs
from copy import deepcopy
from osb3d.OSB3D.Python.env.osb3d_env import OSB3DEnv


class OSB3DEnvWrapper:

    def __init__(self, game_name, configuration_file, no_graphics, worker_id, max_episode_timesteps, pos_already_normed=True):
        self.no_graphics = no_graphics
        self.env = OSB3DEnv( game_name=game_name,
                worker_id=worker_id,
                no_graphics=no_graphics,
                seed=1337,
                config_file=configuration_file,
                max_episode_timestep=max_episode_timesteps,)
        self.env.reset()
        self._max_episode_timesteps = max_episode_timesteps

        self.config = None
        # Defined the values to sample for goal-conditioned policy
        self.reward_weights = None


    def execute(self, actions, visualize=False):
        state, reward, done, _, info = self.env.step(actions)
        state = self.prepare_state(state)
        state = dict(global_in=state)
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])
        return state, done, reward, info

    def reset(self):

        # Sample a motivation reward weight
        self.reward_weights = self.config['reward_weights']
        self.win_weight = self.config['win_weight']
        self.sample_weights = self.reward_weights[np.random.randint(len(self.reward_weights))]
        self.sample_win = self.win_weight[np.random.randint(len(self.win_weight))]
        self.sample_weights = [self.sample_win, self.sample_weights, 1-self.sample_weights]

        # Here the curriculum
        state, info = self.env.reset()
        state = self.prepare_state(state)
        state = dict(global_in=state)

        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])
        return state

    def prepare_state(self, state):
        # The state will come from the env as a list of np arrays
        # Here we just do a list concatenation of all the arrays
        # We will reshape them in the model
        state = [np.asarray(ob).reshape(-1) for ob in state]
        state = np.concatenate(state)
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))

        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.env.close()