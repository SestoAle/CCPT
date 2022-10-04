import os
import numpy as np
import json
from utils import NumpyEncoder
import time

class Runner:
    def __init__(self, agent, frequency, env, save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None, curriculum_mode='steps', evaluation=False,
                 callback_function = None, motivation=None,
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
        self.env = env
        self.curriculum_mode = curriculum_mode
        self.evaluation = evaluation

        # TODO: pass this as an argument
        self.motivation_frequency = 1

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
            "env_rewards": []
        }

        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                # Before asking for demonstrations, set the curriculum of the environment
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                self.env.set_config(config)
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env, dems_name=dems_name)
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env, with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                # Set demonstrations for the environment
                # self.env.set_demonstrations(dems)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                #print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(env, self.reward_frequency, random=True)
                self.reward_model.train()

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


        # Decaying weight of the motivation/inverse reinforcement learning model
        self.last_episode_for_decaying = 0
        # if self.motivation is not None:
        #     self.motivation.motivation_weight = 0.8
        #     self.min_motivation_weight = 0.2

    def run(self):

        # Trainin loop
        # Start training
        start_time = time.time()
        while self.ep <= self.total_episode:
            # Reset the episode
            self.ep += 1
            step = 0

            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
            if self.start_training == 0:
                print(config)
            self.start_training = 1
            self.env.set_config(config)

            state = self.env.reset()
            done = False
            # Total reward of the episode
            episode_reward = 0
            # Total reward of the environment, in case of IRL it can be different from the actual reward of the agent
            env_episode_reward = 0

            # Save local entropies
            local_entropies = []

            # If recurrent, initialize hidden state
            if self.recurrent:
                internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

            # Episode loop
            while True:

                # Evaluation - Execute step
                if not self.recurrent:
                    action, logprob, probs = self.agent.eval([state])

                else:
                    action, logprob, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal, v_internal)

                # If we want to test the agent, we take the highest probability action instead of the sampled
                # one output by eval()
                if self.evaluation:
                    action = [np.argmax(probs)]

                if self.random_actions is not None and self.total_step < self.random_actions:
                    action = [np.random.randint(self.agent.action_size)]

                action = action[0]
                visualize = False
                # Manual input
                # action = 99
                # while (action == 99):
                #     action = input(': ')
                #     if action == 'v':
                #         visualize = True
                #     action = self.env.command_to_action(action)
                # Save probabilities for entropy
                local_entropies.append(self.env.entropy(probs[0]))

                # Execute in the environment
                state_n, done, reward = self.env.execute(action, visualize)

                # Intrinsic Motivation
                # Add the next state to the motivation buffer
                # - The intrinsic reward will be added later -
                if self.motivation is not None:

                    motivation_reward = self.motivation.eval([state_n])
                    print(motivation_reward)
                    self.motivation.add_to_buffer(state_n)

                # Inverse Reinforcement Learning
                # Add the next state to the IRL buffer
                # - The intrinsic reward will be added later -
                if self.reward_model is not None:

                    irl_reward = self.reward_model.eval([state], [state_n],
                                                            [action])
                    print(irl_reward)
                    self.reward_model.add_to_policy_buffer([state], [state_n], [action])

                # If step is equal than max timesteps, terminate the episode
                if step >= self.env._max_episode_timesteps - 1:
                    done = True
                # Time horizon
                elif self.frequency_mode == 'timesteps' and (self.total_step + 1) % self.frequency == 0:
                    done = 2

                # Get the cumulative reward
                episode_reward += reward

                # Update PPO memory
                if not self.recurrent:
                    self.agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                else:
                    try:
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 internal.c[0], internal.h[0], v_internal.c[0], v_internal.h[0])
                    except Exception as e:
                        zero_state = np.reshape(internal[0], [-1,])
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 zero_state, zero_state, zero_state, zero_state)
                    internal = internal_n
                    v_internal = v_internal_n
                state = state_n

                step += 1
                self.total_step += 1

                # If frequency timesteps are passed, update the policy
                if not self.evaluation and self.frequency_mode == 'timesteps' and \
                        self.total_step > 0 and self.total_step % self.frequency == 0:
                    if self.random_actions is not None:
                        if self.total_step > self.random_actions:
                            self.agent.train()
                    else:

                        if self.motivation is not None:
                            # TODO: move this into a function
                            # Normalize observation of the motivation buffer
                            # self.motivation.normalize_buffer()
                            # Compute intrinsic rewards
                            intrinsic_rews = self.motivation.eval(self.agent.buffer['states_n'])

                            # Normalize rewards
                            intrinsic_rews -= np.mean(intrinsic_rews)
                            intrinsic_rews /= np.std(intrinsic_rews)
                            intrinsic_rews *= self.motivation.motivation_weight
                            self.agent.buffer['rewards'] = list(
                                intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                        if self.reward_model is not None:
                            # Compute intrinsic rewards
                            intrinsic_rews = self.reward_model.eval(self.agent.buffer['states'],
                                                                    self.agent.buffer['states_n'],
                                                                    self.agent.buffer['actions'])

                            # Normalize rewards
                            intrinsic_rews -= np.mean(intrinsic_rews)
                            intrinsic_rews /= np.std(intrinsic_rews)
                            intrinsic_rews *= self.reward_model.reward_model_weight
                            self.agent.buffer['rewards'] = list(
                                intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                        # Train the agent
                        self.agent.train()

                    # If frequency episodes are passed, update the policy
                    if not self.evaluation and self.frequency_mode == 'timesteps' and \
                            self.total_step > 0 and self.total_step % self.motivation_frequency == 0:

                        # If we use intrinsic motivation, update also intrinsic motivation
                        if self.motivation is not None:
                            self.update_motivation()

                    # If frequency episodes are passed, update the policy
                    if not self.evaluation and self.frequency_mode == 'timesteps' and \
                            self.total_step > 0 and self.total_step % self.reward_frequency == 0:

                        # If we use intrinsic motivation, update also intrinsic motivation
                        if self.reward_model is not None and not self.fixed_reward_model:
                            self.update_reward_model()

                # If done, end the episode and save statistics
                if done == 1:
                    self.history['episode_rewards'].append(episode_reward)
                    self.history['episode_timesteps'].append(step)
                    self.history['mean_entropies'].append(np.mean(local_entropies))
                    self.history['std_entropies'].append(np.std(local_entropies))
                    self.history['env_rewards'].append(env_episode_reward)
                    break

            # Logging information
            if self.ep > 0 and self.ep % self.logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                if self.reward_model is not None:
                    print('Mean of {} environment episode reward after {} episodes: {}'.
                            format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                print('The agent made a total of {} steps'.format(self.total_step))

                if self.callback_function is not None:
                    self.callback_function(self.agent, self.env, self)

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

                if self.motivation is not None:
                    # Normalize observation of the motivation buffer
                    # self.motivation.normalize_buffer()
                    # Compute intrinsic rewards
                    intrinsic_rews = self.motivation.eval(self.agent.buffer['states_n'])

                    # Normalize rewards
                    # intrinsic_rews -= self.motivation.r_norm.mean
                    # intrinsic_rews /= self.motivation.r_norm.std
                    intrinsic_rews -= np.mean(intrinsic_rews)
                    intrinsic_rews /= np.std(intrinsic_rews)
                    intrinsic_rews *= self.motivation.motivation_weight
                    if self.alternate_frequency > 0:
                        if self.alternate_turn == 0:
                            self.agent.buffer['rewards'] = list(intrinsic_rews)
                    else:
                        self.agent.buffer['rewards'] = list(intrinsic_rews)

                if self.reward_model is not None:

                    # Compute intrinsic rewards
                    intrinsic_rews = self.reward_model.eval(self.agent.buffer['states'], self.agent.buffer['states_n'],
                                                            self.agent.buffer['actions'])

                    # Normalize rewards
                    # intrinsic_rews -= self.reward_model.r_norm.mean
                    # intrinsic_rews /= self.reward_model.r_norm.std

                    #intrinsic_rews = (intrinsic_rews - np.min(intrinsic_rews)) / (np.max(intrinsic_rews) - np.min(intrinsic_rews))
                    intrinsic_rews -= np.mean(intrinsic_rews)
                    intrinsic_rews /= np.std(intrinsic_rews)
                    if self.last_episode_for_decaying > 0:
                        intrinsic_rews *= (1 - self.motivation.motivation_weight)
                    else:
                        intrinsic_rews *= self.reward_model.reward_model_weight

                    if self.alternate_frequency > 0:
                        if self.alternate_turn == 1:
                            self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))
                    else:
                        self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                self.agent.train()
                # For alternating between motivation and imitation learning
                if self.alternate_frequency > 0:
                    self.alternate_count += 1
                    if self.alternate_count % self.alternate_frequency == 0:
                        self.alternate_turn = (self.alternate_turn + 1) % 2

            # If frequency episodes are passed, update the policy
            if not self.evaluation and self.frequency_mode == 'episodes' and \
                    self.ep > 0 and self.ep % self.reward_frequency == 0:

                # If we use intrinsic motivation, update also intrinsic motivation
                if self.reward_model is not None and not self.fixed_reward_model:
                    self.update_reward_model()


            # Save model and statistics
            if self.ep > 0 and self.ep % self.save_frequency == 0:
                self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)

            # Decaying the motivation weight
            if self.last_episode_for_decaying > 0:
                if self.ep < self.last_episode_for_decaying:
                    self.motivation.motivation_weight -= ((0.8 - self.min_motivation_weight) / self.last_episode_for_decaying)


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

        # Save the tf model
        agent.save_model(name=model_name, folder='saved')

        # If we use intrinsic motivation, save the motivation model
        if self.motivation is not None:
            self.motivation.save_model(name=model_name, folder='saved')

        # If we use IRL, save the reward model
        if self.reward_model is not None and not self.fixed_reward_model:
            self.reward_model.save_model('{}_{}'.format(model_name, self.ep))

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
                self.agent.save_model(name=self.agent.model_name + '_' + str(curriculum_step), folder='saved/adversarial')
                # Load the weights of the current version of the main agent to the double agent
                self.double_agent.load_model(name=self.agent.model_name + '_' + str(curriculum_step), folder='saved/adversarial')

        self.current_curriculum_step = curriculum_step

        return config

    # Update intrinsic motivation
    # Update its statistics AND train the model. We print also the model loss
    def update_motivation(self):
        loss = self.motivation.train()
        #print('Mean motivation loss = {}'.format(loss))

    # Update reward model
    # Update its statistics AND train the model. We print also the model loss
    def update_reward_model(self):
        loss, _ = self.reward_model.train()
        #print('Mean reward loss = {}'.format(loss))

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
                state_n, terminal, step_reward = env.execute(actions=action)
                self.reward_model.add_to_policy_buffer([state], [state_n], [action])

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
