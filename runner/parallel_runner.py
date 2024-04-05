import os
import numpy as np
import json
from utils import NumpyEncoder
import time
from threading import Thread
import sys
from copy import deepcopy

# Act thread
class ActThreaded(Thread):
    def __init__(self, agent, env, parallel_buffer, index, config, num_steps, states, recurrent=False, internals=None,
                 v_internals=None, motivation=None, reward_model=None):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.index = index
        self.env.set_config(config)
        self.num_steps = num_steps
        self.agent = agent
        self.states = states
        self.internals = internals
        self.v_internals = v_internals
        self.recurrent = recurrent
        self.motivation = motivation
        self.reward_model = reward_model
        super().__init__()


    def run(self) -> None:
        state = self.states[self.index]
        if self.recurrent:
            internal = self.internals[self.index]
            v_internal = self.v_internals[self.index]
        total_reward = 0
        step = 0
        entropies = []
        for i in range(self.num_steps):
            # Execute the environment with the action
            if not self.recurrent:
                actions, logprobs, probs = self.agent.eval([state])
            else:
                actions, logprobs, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal,
                                                                                               v_internal)
            entropies.append(self.env.entropy(probs[0]))
            actions = actions[0]
            state_n, done, reward, info = self.env.execute(actions)
            step += 1
            total_reward += reward

            self.parallel_buffer['states'][self.index].append(state)
            self.parallel_buffer['states_n'][self.index].append(state_n)
            self.parallel_buffer['done'][self.index].append(done)
            self.parallel_buffer['reward'][self.index].append(reward)
            self.parallel_buffer['action'][self.index].append(actions)
            self.parallel_buffer['logprob'][self.index].append(logprobs)
            if self.recurrent:
                self.parallel_buffer['internal'][self.index].append(internal)
                self.parallel_buffer['v_internal'][self.index].append(v_internal)
                internal = internal_n
                v_internal = v_internal_n

            if self.motivation:
                self.parallel_buffer['motivation'][self.index]['state_n'].append(state_n)

            if self.reward_model:
                self.parallel_buffer['reward_model'][self.index]['state'].append(state)
                self.parallel_buffer['reward_model'][self.index]['state_n'].append(state_n)
                self.parallel_buffer['reward_model'][self.index]['action'].append(actions)

            state = state_n
            if done:
                state = self.env.reset()
                if self.recurrent:
                    internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

        if not done:
            self.parallel_buffer['done'][self.index][-1] = 2

        self.parallel_buffer['episode_rewards'][self.index].append(total_reward)
        self.parallel_buffer['episode_timesteps'][self.index].append(self.num_steps)
        self.parallel_buffer['mean_entropies'][self.index].append(0)
        self.parallel_buffer['std_entropies'][self.index].append(0)
        self.states[self.index] = state
        if self.recurrent:
            self.internals[self.index] = internal
            self.v_internals[self.index] = v_internal

# Epsiode thread
class EpisodeThreaded(Thread):
    def __init__(self, env, parallel_buffer, agent, index, config, num_episode=1, recurrent=False, motivation=False,
                 reward_model=False):
        self.env = env
        self.parallel_buffer = parallel_buffer
        self.agent = agent
        self.index = index
        self.num_episode = num_episode
        self.recurrent = recurrent
        self.env.set_config(config)
        self.motivation = motivation
        self.reward_model = reward_model
        super().__init__()

    def run(self) -> None:
        # Run each thread for num_episode episodes

        for i in range(self.num_episode):
            done = False
            step = 0
            # Reset the environment
            state = self.env.reset()

            # Total episode reward
            episode_reward = 0

            # Local entropies of the episode
            local_entropies = []

            # If recurrent, initialize hidden state
            if self.recurrent:
                internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

            while not done:
                # Evaluation - Execute step
                if not self.recurrent:
                    actions, logprobs, probs = self.agent.eval([state])
                else:
                    actions, logprobs, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal,
                                                                                                 v_internal)
                actions = actions[0]
                state_n, done, reward, info = self.env.execute(actions)

                #reward = reward[0]
                #done = done[0]

                episode_reward += reward
                local_entropies.append(self.env.entropy(probs[0]))
                # If step is equal than max timesteps, terminate the episode
                if step >= self.env._max_episode_timesteps - 1:
                    done = True
                self.parallel_buffer['states'][self.index].append(state)
                self.parallel_buffer['states_n'][self.index].append(state_n)
                self.parallel_buffer['done'][self.index].append(done)
                self.parallel_buffer['reward'][self.index].append(reward)
                self.parallel_buffer['action'][self.index].append(actions)
                self.parallel_buffer['logprob'][self.index].append(logprobs)

                if self.recurrent:
                    self.parallel_buffer['internal'][self.index].append(internal)
                    self.parallel_buffer['v_internal'][self.index].append(v_internal)
                    internal = internal_n
                    v_internal = v_internal_n

                if self.motivation:
                    self.parallel_buffer['motivation'][self.index]['state_n'].append(state_n)

                if self.reward_model:
                    self.parallel_buffer['reward_model'][self.index]['state'].append(state)
                    self.parallel_buffer['reward_model'][self.index]['state_n'].append(state_n)
                    self.parallel_buffer['reward_model'][self.index]['action'].append(actions)

                state = state_n
                step += 1

            # History statistics
            self.parallel_buffer['episode_rewards'][self.index].append(episode_reward)
            self.parallel_buffer['episode_timesteps'][self.index].append(step)
            self.parallel_buffer['mean_entropies'][self.index].append(np.mean(local_entropies))
            self.parallel_buffer['std_entropies'][self.index].append(np.std(local_entropies))
            self.parallel_buffer['info'][self.index].append(info)



class Runner:
    def __init__(self, agent, frequency, envs, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None, curriculum_mode='steps', evaluation=False,
                 callback_function=None, motivation=None,
                 # IRL
                 reward_model=None, fixed_reward_model=False, dems_name='', reward_frequency=30,
                 # Adversarial Play
                 adversarial_play=False, double_agent=None,
                 **kwargs):

        # Runner objects and parameters
        self.agent = agent
        self.curriculum = curriculum
        self.total_episode = total_episode
        self.frequency = frequency
        self.frequency_mode = frequency_mode
        self.random_actions = random_actions
        self.logging = logging
        self.save_frequency = save_frequency
        self.envs = envs
        self.curriculum_mode = curriculum_mode
        self.evaluation = evaluation

        # TODO: pass this as an argument
        self.motivation_frequency = 10

        # For alternating between motivation and imitation reward
        self.alternate_frequency = 0
        self.alternate_count = 0
        self.alternate_turn = 0

        # If we want to use intrinsic motivation
        # Right now only RND is available
        self.motivation = motivation

        # Function to call at the end of each episode.
        # It takes the agent, the runner and the env as input arguments
        self.callback_function = callback_function

        # Recurrent
        self.recurrent = self.agent.recurrent

        # Objects and parameters for IRL
        self.reward_model = reward_model
        self.fixed_reward_model = fixed_reward_model
        self.dems_name = dems_name
        self.reward_frequency = reward_frequency

        # Adversarial play
        self.adversarial_play = adversarial_play
        self.double_agent = double_agent
        # If adversarial play, save the first version of the main agent and load it to the double agent
        if self.adversarial_play:
            self.agent.save_model(name=self.agent.model_name + '_0', folder='saved/adversarial')
            self.double_agent.load_model(name=self.agent.model_name + '_0', folder='saved/adversarial')

        # Global runner statistics
        # total episode
        self.ep = 0
        # total steps
        self.total_step = 0
        # Initialize history
        # History to save model statistics
        self.history = {
            "episode_rewards": [],
            "episode_timesteps": [],
            "mean_entropies": [],
            "std_entropies": [],
            "reward_model_loss": [],
            "env_rewards": [],
            "info": []
        }

        # Initialize parallel buffer for savig experience of each thread without race conditions
        self.parallel_buffer = None
        self.parallel_buffer = self.clear_parallel_buffer()

        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                # Before asking for demonstrations, set the curriculum of the environment
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                self.envs[0].set_config(config)
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0])
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.envs[0], with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                # print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # # Set demonstrations for the environment
                # for e in self.games:
                #     e.set_demonstrations(dems)

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(self.envs[0], self.reward_frequency, random=True)
                self.update_reward_model()
        # elif self.motivation is not None:
        #     # If there is only intrinsic motivation, do some episode for the normalization buffer
        #     self.envs[0].set_config(config)
        #     self.get_experience(self.envs[0], self.motivation_frequency, random=True)

        # For curriculum training
        self.start_training = 0
        self.current_curriculum_step = 0

        # If a saved model with the model_name already exists, load it (and the history attached to it)
        if os.path.exists('{}/{}.meta'.format('saved', agent.model_name)):
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("There's already an agent saved with name {}, "
                               "do you want to continue training? [y/n] ".format(agent.model_name))

            if answer == 'y':
                self.history = self.load_model(agent.model_name, agent)
                self.ep = len(self.history['episode_timesteps'])
                self.total_step = np.sum(self.history['episode_timesteps'])

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 episode
    def create_episode_threads(self, parallel_buffer, agent, config):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(EpisodeThreaded(env=e, index=i, agent=agent, parallel_buffer=parallel_buffer, config=config,
                                           recurrent=self.recurrent, motivation=(self.motivation is not None),
                                           reward_model=(self.reward_model is not None)))

        # Return threads
        return threads

    # Return a list of thread, that will save the experience in the shared buffer
    # The thread will run for 1 step of the environment
    def create_act_threds(self, agent, parallel_buffer, config, states, num_steps, internals=None, v_internals=None):
        # The number of thread will be equal to the number of environments
        threads = []
        for i, e in enumerate(self.envs):
            # Create a thread
            threads.append(ActThreaded(agent=agent, env=e, index=i, parallel_buffer=parallel_buffer, config=config,
                                       states=states, num_steps=num_steps, recurrent=self.recurrent,
                                       internals=internals, v_internals=v_internals,
                                       motivation=(self.motivation is not None),
                                       reward_model=(self.reward_model is not None)))

        # Return threads
        return threads

    # Clear parallel buffer to avoid memory leak
    def clear_parallel_buffer(self):
        # Manually delete parallel buffer
        if self.parallel_buffer is not None:
            del self.parallel_buffer
        # Initialize parallel buffer for savig experience of each thread without race conditions
        parallel_buffer = {
            'states': [],
            'states_n': [],
            'done': [],
            'reward': [],
            'action': [],
            'logprob': [],
            'internal': [],
            'v_internal': [],
            # Motivation
            'motivation': [],
            # Reward model
            'reward_model': [],
            # History
            'episode_rewards': [],
            'episode_timesteps': [],
            'mean_entropies': [],
            'std_entropies': [],
            'info': [],
        }

        for i in range(len(self.envs)):
            parallel_buffer['states'].append([])
            parallel_buffer['states_n'].append([])
            parallel_buffer['done'].append([])
            parallel_buffer['reward'].append([])
            parallel_buffer['action'].append([])
            parallel_buffer['logprob'].append([])
            parallel_buffer['internal'].append([])
            parallel_buffer['v_internal'].append([])
            # Motivation
            parallel_buffer['motivation'].append(
                dict(state_n=[])
            )
            # Reward Model
            parallel_buffer['reward_model'].append(
                dict(state=[], state_n=[], action=[])
            )

            # History
            parallel_buffer['episode_rewards'].append([])
            parallel_buffer['episode_timesteps'].append([])
            parallel_buffer['mean_entropies'].append([])
            parallel_buffer['std_entropies'].append([])

            parallel_buffer['info'].append([])


        return parallel_buffer

    def run(self):

        # Trainin loop
        # Start training
        start_time = time.time()
        # If parallel act is in use, reset all environments at beginning of training

        if self.frequency_mode == 'timesteps':
            states = []
            if self.recurrent:
                internals = []
                v_internals = []
            for env in self.envs:
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                env.set_config(config)
                states.append(env.reset())
                if self.recurrent:
                    internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                    internals.append(internal)
                    v_internals.append(v_internal)

        while self.ep <= self.total_episode:
            # Reset the episode
            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
            if self.start_training == 0:
                print(config)
            self.start_training = 1

            # Episode loop
            if self.frequency_mode=='episodes':
            # If frequency is episode, run the episodes in parallel
                # Create threads
                threads = self.create_episode_threads(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config)

                # Run the threads
                for t in threads:
                    t.start()

                # Wait for the threads to finish
                for t in threads:
                    t.join()

                self.ep += len(threads)
            else:
            # If frequency is timesteps, run only the 'execute' in parallel for horizon steps
                # Create threads
                if self.recurrent:
                    threads = self.create_act_threds(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config,
                                                 states=states, num_steps=self.frequency, internals=internals, v_internals=v_internals)
                else:
                    threads = self.create_act_threds(agent=self.agent, parallel_buffer=self.parallel_buffer, config=config,
                                                 states=states, num_steps=self.frequency)

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()

                # Delete threads from memory
                del threads[:]

                # Get how many episodes and steps are passed within threads
                self.ep += np.sum(np.asarray(self.parallel_buffer['done'][:]) == 1)
                self.total_step += len(self.envs) * self.frequency

            # Add the overall experience to the buffer
            # Update the history
            for i in range(len(self.envs)):

                if not self.recurrent:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i]
                    ):
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                else:
                    # Add to the agent experience in order of execution
                    for state, state_n, action, reward, logprob, done, internal, v_internal in zip(
                            self.parallel_buffer['states'][i],
                            self.parallel_buffer['states_n'][i],
                            self.parallel_buffer['action'][i],
                            self.parallel_buffer['reward'][i],
                            self.parallel_buffer['logprob'][i],
                            self.parallel_buffer['done'][i],
                            self.parallel_buffer['internal'][i],
                            self.parallel_buffer['v_internal'][i],
                    ):
                        try:
                            self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     internal.c[0], internal.h[0], v_internal.c[0], v_internal.h[0])
                        except Exception as e:
                            zero_state = np.reshape(internal[0], [-1, ])
                            self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                     zero_state, zero_state, zero_state, zero_state)

                # For motivation, add the agents experience to the motivation buffer
                if self.motivation is not None:
                    for state_n in self.parallel_buffer['motivation'][i]['state_n']:
                        # We need deepcopy because the state will be normalized (and not for the policy)
                        self.motivation.add_to_buffer(deepcopy(state_n))

                if self.reward_model is not None:
                    self.reward_model.add_to_policy_buffer(self.parallel_buffer['reward_model'][i]['state'],
                                                       self.parallel_buffer['reward_model'][i]['state_n'],
                                                       self.parallel_buffer['reward_model'][i]['action'])

                # Upadte the hisotry in order of execution
                for episode_reward, step, mean_entropies, std_entropies, info in zip(
                        self.parallel_buffer['episode_rewards'][i],
                        self.parallel_buffer['episode_timesteps'][i],
                        self.parallel_buffer['mean_entropies'][i],
                        self.parallel_buffer['std_entropies'][i],
                        self.parallel_buffer['info'][i],
                ):
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(mean_entropies)
                    self.history['std_entropies'].append(std_entropies)
                    self.history['info'].append(info)


            # Clear parallel buffer
            self.parallel_buffer = self.clear_parallel_buffer()

            # If frequency timesteps are passed, update the motivation
            if not self.evaluation and self.frequency_mode == 'timesteps' and \
                    self.total_step > 0 and self.total_step % (self.motivation_frequency * len(self.envs)) == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.motivation is not None:
                    self.update_motivation()

            # If frequency timesteps are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'timesteps' and \
                    self.total_step > 0 and self.total_step % (self.frequency * len(self.envs)) == 0:

                # Compute intrinsic rewards (if any)
                self.compute_intrinsic_rewards()

                self.agent.train()

            # If frequency timesteps are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'timesteps' and \
                    self.total_step > 0 and self.total_step % (self.reward_frequency * len(self.envs)) == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.reward_model is not None and not self.fixed_reward_model:
                    self.update_reward_model()

            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                if self.reward_model is not None:
                    print('Mean of {} environment episode reward after {} episodes: {}'.
                            format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(np.sum(self.history['episode_timesteps'])))

                if self.callback_function is not None:
                    self.callback_function(self.agent, self.envs, self)

                self.timer(start_time, time.time())

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.motivation_frequency == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.motivation is not None:
                    self.update_motivation()

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.frequency == 0:

                if self.random_actions is not None:
                    if self.total_step <= self.random_actions:
                        self.motivation.clear_buffer()
                        continue

                # Compute intrinsic rewards (if any)
                self.compute_intrinsic_rewards()

                self.agent.train()

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.reward_frequency == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.reward_model is not None and not self.fixed_reward_model:
                    self.update_reward_model()

            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)

    def save_model(self, history, model_name, curriculum, agent):

        # Save statistics as json
        json_str = json.dumps(history, cls=NumpyEncoder)
        f = open("arrays/{}.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save curriculum as json
        json_str = json.dumps(curriculum, cls=NumpyEncoder)
        f = open("arrays/{}_curriculum.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save statistics of RND (mean and std) as json
        if self.motivation is not None:
            stats = dict(mean=self.motivation.obs_norm.mean, std=self.motivation.obs_norm.mean)
            json_str = json.dumps(stats, cls=NumpyEncoder)
            f = open("arrays/{}_rnd_stat.json".format(model_name), "w")
            f.write(json_str)
            f.close()

        # Save the tf model
        agent.save_model(name=model_name, folder='saved')

        # If we use intrinsic motivation, save the motivation model
        if self.motivation is not None:
            self.motivation.save_model(name=model_name, folder='saved')

        # If we use IRL, save the reward model
        if self.reward_model is not None and not self.fixed_reward_model:
            self.reward_model.save_model('{}'.format(model_name))

        print('Model saved with name: {}'.format(model_name))

    def load_model(self, model_name, agent):
        agent.load_model(name=model_name, folder='saved')

        # Load intrinsic motivation for testing
        if self.motivation is not None:
            self.motivation.load_model(name=model_name, folder='saved')

        # # Load reward motivation for testing
        # if self.reward_model is not None:
        #     self.reward_model.load_model(name=model_name, folder='saved')

        with open("arrays/{}.json".format(model_name)) as f:
            history = json.load(f)

        return history

    # Update curriculum for DeepCrawl
    def set_curriculum(self, curriculum, history, mode='steps'):

        total_timesteps = np.sum(history['episode_timesteps'])
        total_episodes = len(history['episode_timesteps'])

        if curriculum == None:
            return None

        if mode == 'episodes':
            lessons = np.cumsum(curriculum['thresholds'])
            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_episodes > l:
                    curriculum_step = index + 1

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_timesteps > l:
                    curriculum_step = index + 1

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[curriculum_step]

        # If Adversarial play
        if self.adversarial_play:
            if curriculum_step > self.current_curriculum_step:
                # Save the current version of the main agent
                self.agent.save_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                      folder='saved/adversarial')
                # Load the weights of the current version of the main agent to the double agent
                self.double_agent.load_model(name=self.agent.model_name + '_' + str(curriculum_step),
                                             folder='saved/adversarial')

        self.current_curriculum_step = curriculum_step

        return config

    # For IRL, get initial experience from environment, the agent act in the env without update itself
    def get_experience(self, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.frequency

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            state = env.reset()
            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                if random:
                    num_actions = self.agent.action_size
                    action = np.random.randint(0, num_actions)
                else:
                    action, _, c_probs = self.agent.eval([state])
                state_n, terminal, step_reward, info = env.execute(actions=action)
                if self.reward_model is not None:
                    self.reward_model.add_to_policy_buffer([state], [state_n], [action])

                if self.motivation is not None:
                    self.motivation.add_to_buffer(deepcopy(state_n))

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if self.motivation is not None:
                self.motivation.clear_buffer()

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Update intrinsic motivation
    # Update its statistics AND train the model. We print also the model loss
    def update_motivation(self):
        loss = self.motivation.train()
        # print('Mean motivation loss = {}'.format(loss))

    # Update reward model
    # Update its statistics AND train the model. We print also the model loss
    def update_reward_model(self):
        loss, _ = self.reward_model.train()
        self.history['reward_model_loss'].append(loss)

        # if len(self.agent.buffer['states']) > 0:
        #     self.reward_model_std = np.std(self.reward_model.eval(self.agent.buffer['states'], self.agent.buffer['states_n'],
        #                                                     self.agent.buffer['actions']))
        #
        #     self.reward_model_mean = np.mean(self.reward_model.eval(self.agent.buffer['states'], self.agent.buffer['states_n'],
        #                            self.agent.buffer['actions']))
        # else:
        #     self.reward_model_mean = 0
        #     self.reward_model_std = 1

        print('Mean reward loss = {}'.format(loss))


    def compute_intrinsic_rewards(self):
        if self.motivation is not None:
            # Normalize observation of the motivation buffer
            # self.motivation.normalize_buffer()
            # Compute intrinsic rewards
            # intrinsic_rews = self.motivation.eval(self.agent.buffer['states_n'])
            # Compute intrinsic rewards
            num_batches = 10
            batch_size = int(np.ceil(len(self.agent.buffer['states_n']) / num_batches))
            intrinsic_rews = []
            for i in range(num_batches):
                c_intrinsic_rews = self.motivation.eval(deepcopy(
                    self.agent.buffer['states_n'][batch_size * i:batch_size * i + batch_size]))
                # for rew in c_intrinsic_rews:
                #     self.reward_model.r_norm.push(rew)
                intrinsic_rews.extend(list(c_intrinsic_rews))

            # Get the weights of the motivation rewards
            weights = [state['global_in'][-3:] for state in
                       self.agent.buffer['states']]
            weights = np.asarray(weights)
            # Normalize rewards
            # intrinsic_rews -= self.motivation.r_norm.mean
            # intrinsic_rews /= self.motivation.r_norm.std
            intrinsic_rews -= np.mean(intrinsic_rews)
            intrinsic_rews /= (np.std(intrinsic_rews) + 1e-5)
            intrinsic_rews *= self.motivation.motivation_weight

            # Weight of win
            self.agent.buffer['rewards'] = np.asarray(self.agent.buffer['rewards']) * weights[:, 0]

            intrinsic_rews *= weights[:, 1]
            self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))
            print(self.agent.buffer['rewards'])

        if self.reward_model is not None:

            # Compute intrinsic rewards
            num_batches = 10
            batch_size = int(np.ceil(len(self.agent.buffer['states']) / num_batches))
            intrinsic_rews = []
            for i in range(num_batches):
                c_intrinsic_rews = self.reward_model.eval(
                    self.agent.buffer['states'][batch_size * i:batch_size * i + batch_size],
                    self.agent.buffer['states_n'][batch_size * i:batch_size * i + batch_size],
                    self.agent.buffer['actions'][batch_size * i:batch_size * i + batch_size])
                # for rew in c_intrinsic_rews:
                #     self.reward_model.r_norm.push(rew)
                intrinsic_rews.extend(list(c_intrinsic_rews))

            intrinsic_rews = np.asarray(intrinsic_rews)
            # Normalize rewards
            # intrinsic_rews -= self.reward_model.r_norm.mean
            # intrinsic_rews /= (self.reward_model.r_norm.std + 1e-5)

            # Get the weights of the motivation rewards
            weights = [state['global_in'][-3:] for state in
                       self.agent.buffer['states']]
            weights = np.asarray(weights)
            # intrinsic_rews = (intrinsic_rews - np.min(intrinsic_rews)) / (np.max(intrinsic_rews) - np.min(intrinsic_rews))
            intrinsic_rews -= np.mean(intrinsic_rews)
            intrinsic_rews /= (np.std(intrinsic_rews) + 1e-5)
            # intrinsic_rews -= self.reward_model_mean
            # intrinsic_rews /= self.reward_model_std
            intrinsic_rews *= self.reward_model.reward_model_weight
            intrinsic_rews *= weights[:, 2]

            self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
