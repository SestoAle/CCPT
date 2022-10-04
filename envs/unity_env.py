import numpy as np
from mlagents.envs import UnityEnvironment
import matplotlib.pyplot as plt
import logging as logs
from copy import deepcopy

class PlayTestEnvironment:

    def __init__(self, game_name, no_graphics, worker_id, max_episode_timesteps, pos_already_normed=True):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id)
        self.unity_env.reset()
        self._max_episode_timesteps = max_episode_timesteps
        self.default_brain = self.unity_env.brain_names[0]
        self.config = None
        # Table where we save the position for intrinsic reward and spawn position
        self.pos_buffer = dict()
        self.pos_already_normed = pos_already_normed
        self.coverage_of_points = []

        # Dict to store the trajectories at each episode
        self.trajectories_for_episode = dict()
        # Dict to store the actions at each episode
        self.actions_for_episode = dict()
        self.episode = -1

        self.dems = None

        # Defined the values to sample for goal-conditioned policy
        self.reward_weights = None


    def execute(self, actions, visualize=False):
        env_info = self.unity_env.step([actions])[self.default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.actions_for_episode[self.episode].append(actions)

        self.previous_action = actions
        state = dict(global_in=env_info.vector_observations[0])
        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])

        # Get the agent position from the state to compute reward
        position = state['global_in'][:5]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        if visualize:
            threedgrid = np.reshape(state['global_in'][4:4 + 9261], [21, 21, 21])
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            filled = (1 - (threedgrid == 0))
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(threedgrid.min(), threedgrid.max())
            ax.voxels(filled, facecolors=cmap(norm(threedgrid)), edgecolor="black")
            plt.show()

        return state, done, reward

    def reset(self):

        # Sample a motivation reward weight
        self.reward_weights = self.config['reward_weights']
        self.win_weight = self.config['win_weight']
        self.sample_weights = self.reward_weights[np.random.randint(len(self.reward_weights))]
        self.sample_win = self.win_weight[np.random.randint(len(self.win_weight))]
        self.sample_weights = [self.sample_win, self.sample_weights, 1-self.sample_weights]

        if self.dems is not None:
            self.sample_position_from_dems()

        # Change config to be fed to Unity (no list)
        unity_config = dict()
        for key in self.config.keys():
            if key != "reward_weights" and key != 'win_weight':
                unity_config[key] = self.config[key]

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.coverage_of_points.append(len(self.pos_buffer.keys()))
        self.episode += 1
        self.trajectories_for_episode[self.episode] = []
        self.actions_for_episode[self.episode] = []
        # self.set_spawn_position()

        env_info = self.unity_env.reset(train_mode=True, config=unity_config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])
        position = state['global_in'][:5]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # print(np.reshape(state['global_in'][7:7 + 225], [15, 15]))
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))

        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.unity_env.close()

    def multidimensional_shifting(self, num_samples, sample_size, elements, probabilities):
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

    def clear_buffers(self):
        self.trajectories_for_episode = dict()
        self.actions_for_episode = dict()

    def command_to_action(self, command):

        if command == 'w':
            return 3
        if command == 'a':
            return 2
        if command == 's':
            return 4
        if command == 'd':
            return 1

        if command == 'e':
            return 5
        if command == 'c':
            return 7
        if command == 'z':
            return 6
        if command == 'q':
            return 8

        if command == 'r':
            return 0

        if command == ' ':
            return 9

        return 99