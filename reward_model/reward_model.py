import tensorflow as tf
import numpy as np
import pickle
from utils import *
import random
from layers.layers import *

eps = 1e-12


class RewardModel:

    def __init__(self, actions_size, policy, network_architecture, input_architecture, obs_to_state, name, lr,
                 sess=None, buffer_size=100000, gradient_penalty_weight=10.0, reward_model_weight=1.,
                 with_action=False, num_itr=20, batch_size=32, eval_with_probs=False, **kwargs):

        # Initialize some model attributes
        # RunningStat to normalize reward from the model
        self.r_norm = RunningStat()

        # Policy agent needed to compute the discriminator
        self.policy = policy
        # Demonstrations buffer
        self.expert_traj = None
        self.validation_traj = None
        # Policy experience buffer
        self.policy_traj = self.clear_policy_buffer()
        # Num of actions available in the environment
        self.actions_size = actions_size
        # Network structures
        self.input_architecture = input_architecture
        self.network = network_architecture
        self.obs_to_state = obs_to_state
        # If state-action
        self.with_action = with_action
        # Tensorflow name of the model
        self.name = name
        # Tensorflow session
        self.sess = sess
        # Learning rate
        self.lr = lr

        # Model hyper-parameters
        self.eval_with_probs = eval_with_probs
        self.num_itr = num_itr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # Weight of the reward output by the reward model
        self.reward_model_weight = reward_model_weight
        # Gradient penalty
        self.gradient_penalty_weight = gradient_penalty_weight

        # Initialize network architecture
        self.initialize_network(self.input_architecture, self.network)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    # Initialze network architecture
    def initialize_network(self, input, network):
        raise NotImplementedError()

    def train(self):
        raise NotImplemented()

    def eval_discriminator(self, obs, obs_n, probs, acts=None):
        raise NotImplementedError()

    def eval(self, obs, obs_n, acts=None, probs=None):
        raise NotImplementedError()

    # Normalize the reward
    def push_reward(self, rewards):
        for r in rewards:
            self.r_norm.push(r)

    def normalize_rewards(self, rewards):
        rewards -= self.r_norm.mean
        rewards /= (self.r_norm.std + eps)
        rewards *= 0.05

        return rewards

    # Select action from the policy and fetch the probability distribution over the action space
    def select_action(self, state):
        # TODO: eval_recurrent
        act, _, probs = self.policy.eval(state)
        return (act, probs[0])

    # Update demonstrations
    def set_demonstrations(self, demonstrations, validations):
        self.expert_traj = demonstrations

        if validations is not None:
            self.validation_traj = validations

    # Create demonstrations manually or with a policy
    def create_demonstrations(self, env, save_demonstrations=True, inference=False, verbose=False,
                              with_policy=False, num_episode=3,
                              model_name='model', random=False, dems_name='dems_very_acc_com.pkl', sampled_env=None,
                              ):
        end = False

        # Initialize trajectories buffer
        expert_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'episode_len': []
        }

        val_traj = {
            'obs': [],
            'obs_n': [],
            'acts': [],
            'episode_len': []
        }

        if with_policy is False:
            num_episode = None

        if sampled_env is not None and sampled_env > 0:
            num_episode = sampled_env

        episode = 1

        while not end:
            # Make another demonstration
            print('Demonstration n° ' + str(episode))
            # Reset the environment
            state = env.reset()
            print(state['global_in'])
            print(len(state['global_in']))
            states = [state]
            actions = []
            done = False
            step = 0
            # New sequence of states and actions
            while not done:
                try:
                    # Input the action and save the new state and action
                    step += 1
                    print('Timesteps number {}/{}'.format(step, env._max_episode_timesteps))
                    if verbose:
                        env.print_observation(state)
                    if not with_policy:
                        action = input('action: ')
                        if action == "f":
                            done = True
                            continue
                        while env.command_to_action(action) >= 99:
                            action = input('action: ')
                        action = env.command_to_action(action)
                    else:
                        if random:
                            action = np.random.randint(0, self.actions_size)
                        else:
                            action, probs = self.select_action(state)

                    state_n, done, reward = env.execute(action)

                    # If inference is true, print the reward
                    if inference:
                        _, probs = self.select_action(state)
                        reward = self.eval([state], [state_n], [action])
                        _, disc_prob, value, value_n, r_feat, v_feat, vn_feat = self.eval_discriminator([state],
                                                                                                        [state_n],
                                                                                                        [probs[action]],
                                                                                                        [action])

                        print('Discriminator probability: ' + str(disc_prob))
                        print('Unnormalize reward: ' + str(reward))
                        reward = self.normalize_rewards(reward)
                        print('Normalize reward: ' + str(reward))
                        print('Probability of state space: ')
                        print(probs)

                    state = state_n
                    states.append(state)
                    actions.append(action)
                    if step >= env._max_episode_timesteps:
                        done = True
                except Exception as e:
                    print(e)
                    continue

            if not inference:
                y = None
                if with_policy or (sampled_env is not None and sampled_env > 0):
                    if episode <= num_episode:
                        y = 'y'

                while y != 'y' and y != 'n':
                    y = input('Do you want to save this demonstration? [y/n] ')

                if y == 'y':
                    # Update expert trajectories
                    expert_traj['obs'].extend(np.array(states[:-1]))
                    expert_traj['obs_n'].extend(np.array(states[1:]))
                    expert_traj['acts'].extend(np.array(actions))
                    expert_traj['episode_len'].extend(np.asarray([len(states[:-1])]))
                    episode += 1
                else:
                    if num_episode is None or episode >= num_episode:
                        y = input('Do you want to save this demonstration as validation? [y/n] ')
                    else:
                        y = 'n'
                    if y == 'y':
                        val_traj['obs'].extend(np.array(states[:-1]))
                        val_traj['obs_n'].extend(np.array(states[1:]))
                        val_traj['acts'].extend(np.array(actions))
                        val_traj['episode_len'].extend(np.asarray([len(states[:-1])]))
                        episode += 1

            y = None
            if num_episode is None:
                while y != 'y' and y != 'n':
                    if not inference:
                        y = input('Do you want to create another demonstration? [y/n] ')
                    else:
                        y = input('Do you want to try another episode? [y/n] ')

                    if y == 'n':
                        end = True
            else:
                if episode >= num_episode + 1:
                    end = True

        if len(val_traj['obs']) <= 0:
            val_traj = None

        # Save demonstrations to file
        if save_demonstrations and not inference:
            print('Saving the demonstrations...')
            self.save_demonstrations(expert_traj, val_traj, name=dems_name)
            print('Demonstrations saved with name {}!'.format(dems_name))

        if not inference:
            self.set_demonstrations(expert_traj, val_traj)

        return expert_traj, val_traj

    # Save demonstrations dict to file
    @staticmethod
    def save_demonstrations(demonstrations, validations=None, name='dems_acc.pkl'):
        with open('reward_model/dems/' + name, 'wb') as f:
            pickle.dump(demonstrations, f, pickle.HIGHEST_PROTOCOL)
        if validations is not None:
            with open('reward_model/dems/vals_' + name, 'wb') as f:
                pickle.dump(validations, f, pickle.HIGHEST_PROTOCOL)

    # Load demonstrations from file
    def load_demonstrations(self, dems_name):
        with open('reward_model/dems/' + dems_name, 'rb') as f:
            expert_traj = pickle.load(f)

        # with open('reward_model/dems/vals_' + dems_name, 'rb') as f:
        #     val_traj = pickle.load(f)

        val_traj = None

        self.set_demonstrations(expert_traj, val_traj)

        return expert_traj, val_traj

    # Save the entire model
    def save_model(self, name=None):
        tf.compat.v1.disable_eager_execution()
        self.saver.save(self.sess, 'reward_model/models/' + name)
        return

    # Load entire model
    def load_model(self, name=None):
        tf.compat.v1.disable_eager_execution()
        #self.saver = tf.compat.v1.train.import_meta_graph('reward_model/models/' + name + '.meta')
        self.saver.restore(self.sess, 'reward_model/models/' + name)

        print("IL/IRL model loaded!")

        return

    def clear_policy_buffer(self):
        policy_traj = dict()
        policy_traj['obs'] = []
        policy_traj['obs_n'] = []
        policy_traj['acts'] = []
        return policy_traj

    # Add ot policy buffer a new transitions
    def add_to_policy_buffer(self, obs, obs_n, acts, del_mode='random'):
        # The transictions must be a list of transiction

        new_n = len(obs)

        if len(self.policy_traj['obs']) + new_n > self.buffer_size:

            diff = len(self.policy_traj['obs']) + new_n - self.buffer_size

            if del_mode == "latest":
                del self.policy_traj['obs'][0:diff]
                del self.policy_traj['obs_n'][0:diff]
                del self.policy_traj['acts'][0:diff]

            elif del_mode == "prob":

                N = len(self.policy_traj['obs'])
                probs = np.arange(N) + 1
                probs = probs / float(np.sum(probs))
                indices_to_remove = np.random.choice(np.arange(N), diff, p=probs)

                for idx in sorted(indices_to_remove, reverse=True):
                    del self.policy_traj['obs'][idx]
                    del self.policy_traj['obs_n'][idx]
                    del self.policy_traj['acts'][idx]

            elif del_mode == "random":
                indices_to_remove = np.random.choice(len(self.policy_traj['obs']), diff, replace=False)
                for idx in sorted(indices_to_remove, reverse=True):
                    del self.policy_traj['obs'][idx]
                    del self.policy_traj['obs_n'][idx]
                    del self.policy_traj['acts'][idx]

        self.policy_traj['obs'].extend(obs)
        self.policy_traj['obs_n'].extend(obs_n)
        self.policy_traj['acts'].extend(acts)

    # Method that clean the buffer if we want an efficent del_mode='probs'
    def clean_buffer(self):

        overflow = len(self.policy_traj['obs']) - self.buffer_size

        while overflow > 0:
            # self.buffer = self.buffer[overflow:]
            N = len(self.policy_traj['obs'])
            probs = np.arange(N) + 1
            probs = probs / float(np.sum(probs))
            #pidx = multidimensional_shifting(1, 1, np.arange(len(probs)), probs)[0][0]
            #pidx = np.random.choice(np.arange(N), p=probs)
            pidx = random.choices(range(N), weights=probs, k=1)[0]

            del self.policy_traj['obs'][pidx]
            del self.policy_traj['obs_n'][pidx]
            del self.policy_traj['acts'][pidx]

            overflow -= 1


# AIRL model, can be state-only or state-action, with or without value function \phi
class AIRL(RewardModel):

    def __init__(self, actions_size, policy, network, input_architecture, obs_to_state, name='airl', lr=1e-5, sess=None,
                 dems_name='dems',
                 with_action=False, num_itr=20, batch_size=32, eval_with_probs=False, clipping=True,
                 entropy_weight=0.1, with_value=True, gamma=0.9, **kwargs):

        # AIRL specific hyper-parameters
        self.entropy_weight = entropy_weight
        self.clipping = clipping
        self.with_value = with_value
        self.gamma = gamma
        super(AIRL, self).__init__(actions_size, policy, network, input_architecture, obs_to_state, name, lr, sess,
                                   dems_name,
                                   with_action, num_itr=num_itr, batch_size=batch_size, eval_with_probs=eval_with_probs)

    # Initialize network structures
    def initialize_network(self, input_architecture, network):

        with tf.compat.v1.variable_scope(self.name) as vs:
            with tf.compat.v1.variable_scope('irl'):
                self.probs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='probs')
                self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name='labels')

                # Only discrete action space
                # TODO: add continuous action space
                self.act = tf.compat.v1.placeholder(tf.float32, [None, 1], name='act')

                with tf.compat.v1.variable_scope('states'):
                    self.states = input()
                with tf.compat.v1.variable_scope('states_n'):
                    self.states_n = input()

                self.reward = network(self.states, act=self.act, with_action=self.with_action,
                                      actions_size=self.actions_size)
                # Create \phi for disentanglment
                if self.with_value:
                    with tf.compat.v1.variable_scope('value'):
                        self.value = network(self.states, act=self.act, with_action=False)
                    with tf.compat.v1.variable_scope('value', reuse=True):
                        self.value_n = network(self.states_n, act=self.act, with_action=False)

                    self.f = self.reward + self.gamma * self.value_n - self.value
                else:
                    self.f = self.reward
                # Discriminator
                self.discriminator = tf.math.divide(tf.math.exp(self.f), tf.math.add(tf.math.exp(self.f), self.probs))

                # Loss function
                self.loss = -tf.reduce_mean((self.labels * tf.math.log(self.discriminator + eps)) + (
                        (1 - self.labels) * tf.math.log(1 - self.discriminator + eps)))

                if self.clipping:
                    optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
                    gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                    gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 1.0)
                    self.step = optimizer.apply_gradients(zip(gradients, variables))
                else:
                    self.step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sample_method='sample'):
        losses = []

        expert_traj = self.expert_traj
        policy_traj = self.policy_traj

        # Update reward model for num_itr mini-batch steps
        for it in range(self.num_itr):

            if sample_method == 'sample':
                expert_batch_idxs = random.sample(range(len(expert_traj['obs'])), self.batch_size)
                policy_batch_idxs = random.sample(range(len(policy_traj['obs'])), self.batch_size)
            elif sample_method == 'random':
                expert_batch_idxs = np.random.randint(0, len(expert_traj['obs']), self.batch_size)
                policy_batch_idxs = np.random.randint(0, len(policy_traj['obs']), self.batch_size)

            expert_obs = [expert_traj['obs'][id] for id in expert_batch_idxs]
            policy_obs = [policy_traj['obs'][id] for id in policy_batch_idxs]

            expert_obs_n = [expert_traj['obs_n'][id] for id in expert_batch_idxs]
            policy_obs_n = [policy_traj['obs_n'][id] for id in policy_batch_idxs]

            expert_acts = [expert_traj['acts'][id] for id in expert_batch_idxs]
            policy_acts = [policy_traj['acts'][id] for id in policy_batch_idxs]

            expert_probs = []
            for (index, state) in enumerate(expert_obs):
                _, probs = self.select_action(state)
                expert_probs.append(probs[expert_acts[index]])

            policy_probs = []
            for (index, state) in enumerate(policy_obs):
                _, probs = self.select_action(state)
                policy_probs.append(probs[policy_acts[index]])

            expert_probs = np.asarray(expert_probs)
            policy_probs = np.asarray(policy_probs)

            labels = np.ones((self.batch_size, 1))
            labels = np.concatenate([labels, np.zeros((self.batch_size, 1))])

            e_states = self.obs_to_state(expert_obs)
            p_states = self.obs_to_state(policy_obs)

            e_states_n = self.obs_to_state(expert_obs_n)
            p_states_n = self.obs_to_state(policy_obs_n)

            all_states = []
            all_states_n = []

            for i in range(len(e_states)):
                all_states.append(np.concatenate([e_states[i], p_states[i]], axis=0))
                all_states_n.append(np.concatenate([e_states_n[i], p_states_n[i]], axis=0))

            all_probs = np.concatenate([expert_probs, policy_probs], axis=0)
            all_probs = np.expand_dims(all_probs, axis=1)

            feed_dict = {}
            for i in range(len(self.states)):
                feed_dict[self.states[i]] = all_states[i]
                feed_dict[self.states_n[i]] = all_states_n[i]

            feed_dict[self.probs] = all_probs
            feed_dict[self.labels] = labels

            if self.with_action:
                all_acts = np.concatenate([expert_acts, policy_acts], axis=0)
                all_acts = np.expand_dims(all_acts, axis=1)

                feed_dict[self.act] = all_acts

            loss, f, disc, _ = self.sess.run([self.loss, self.f, self.discriminator, self.step],
                                             feed_dict=feed_dict)

            losses.append(loss)

        # Update reward statistics with policy experience buffer just collected
        for it in range(self.num_itr):

            expert_batch_idxs = random.sample(range(len(expert_traj['obs'])), self.batch_size)
            policy_batch_idxs = random.sample(range(len(policy_traj['obs'])), self.batch_size)

            # expert_batch_idxs = np.random.randint(0, len(expert_traj['obs']), self.batch_size)
            # policy_batch_idxs = np.random.randint(0, len(policy_traj['obs']), self.batch_size)

            expert_obs = [expert_traj['obs'][id] for id in expert_batch_idxs]
            policy_obs = [policy_traj['obs'][id] for id in policy_batch_idxs]

            expert_obs_n = [expert_traj['obs_n'][id] for id in expert_batch_idxs]
            policy_obs_n = [policy_traj['obs_n'][id] for id in policy_batch_idxs]

            expert_acts = [expert_traj['acts'][id] for id in expert_batch_idxs]
            policy_acts = [policy_traj['acts'][id] for id in policy_batch_idxs]

            e_states = self.obs_to_state(expert_obs)
            p_states = self.obs_to_state(policy_obs)

            e_states_n = self.obs_to_state(expert_obs_n)
            p_states_n = self.obs_to_state(policy_obs_n)

            all_states = []
            all_states_n = []

            for i in range(len(e_states)):
                all_states.append(np.concatenate([e_states[i], p_states[i]], axis=0))
                all_states_n.append(np.concatenate([e_states_n[i], p_states_n[i]], axis=0))

            expert_probs = []
            for (index, state) in enumerate(expert_obs):
                _, probs = self.select_action(state)
                expert_probs.append(probs[expert_acts[index]])

            policy_probs = []
            for (index, state) in enumerate(policy_obs):
                _, probs = self.select_action(state)
                policy_probs.append(probs[policy_acts[index]])

            expert_probs = np.asarray(expert_probs)
            policy_probs = np.asarray(policy_probs)

            probs = np.concatenate([expert_probs, policy_probs], axis=0)
            probs = np.expand_dims(probs, axis=1)

            feed_dict = {}
            for i in range(len(self.states)):
                feed_dict[self.states[i]] = all_states[i]
                feed_dict[self.states_n[i]] = all_states_n[i]

            if self.with_action:
                all_acts = np.concatenate([expert_acts, policy_acts], axis=0)
                all_acts = np.expand_dims(all_acts, axis=1)

                feed_dict[self.act] = all_acts

            f = self.sess.run([self.f], feed_dict=feed_dict)
            f -= self.entropy_weight * np.log(probs)
            f = np.squeeze(f)
            self.push_reward(f)

        # Update Dynamic Running Stat
        if isinstance(self.r_norm, DynamicRunningStat):
            self.r_norm.reset()

        return np.mean(losses), 0

    def print_obs(self, obs):
        sum = obs[:, :, 0] * 0
        for i in range(1, 7):
            sum += obs[:, :, i] * i
        sum = np.flip(np.transpose(sum), 0)
        return sum

    # Return reward based on f(s,a,s')
    def eval_discriminator(self, obs, obs_n, probs, acts=None):

        states = self.obs_to_state(obs)
        states_n = self.obs_to_state(obs_n)

        probs = np.expand_dims(probs, axis=1)

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.states[i]] = states[i]
            feed_dict[self.states_n[i]] = states_n[i]

        feed_dict[self.probs] = probs

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.act] = acts

        f, disc, value, value_n, = self.sess.run([self.f, self.discriminator, self.value, self.value_n],
                                                 feed_dict=feed_dict)
        f -= self.entropy_weight * np.log(probs)
        f = self.normalize_rewards(f)

        return f, disc, value, value_n

    # Return reward based on r(s,a)
    def eval(self, obs, obs_n, acts=None, probs=None):
        states = self.obs_to_state(obs)
        states_n = self.obs_to_state(obs_n)

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.states[i]] = states[i]
            feed_dict[self.states_n[i]] = states_n[i]

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.act] = acts

        reward = self.sess.run([self.reward], feed_dict=feed_dict)

        if self.eval_with_probs:
            reward -= self.entropy_weight * np.log(probs)

        # reward = self.normalize_rewards(reward)

        return reward[0][0]


# AIRL model, can be state-only or state-action
class GAIL(RewardModel):

    def initialize_network(self, input, network):

        with tf.compat.v1.variable_scope(self.name) as vs:
            with tf.compat.v1.variable_scope('irl'):
                self.labels = tf.compat.v1.placeholder(tf.float32, [None, 1], name='labels')

                self.states, self.act, self.states_n = input()

                with tf.compat.v1.variable_scope('net'):
                    self.logits = network(states=self.states, states_n=self.states_n, act=self.act,
                                                   with_action=self.with_action, actions_size=self.actions_size)
                    self.logits = self.logits[0]

                self.discriminator = tf.nn.sigmoid(self.logits)

                # Loss Function
                # Original loss from GAIL paper
                # self.loss = -tf.reduce_mean((self.labels * tf.math.log(self.discriminator + eps)) + (
                #          (1 - self.labels) * tf.math.log(1 - self.discriminator + eps)))
                # Loss from AMP
                self.loss = tf.reduce_mean((self.labels * tf.math.pow((self.logits - 1), 2)) + (
                      (1 - self.labels) * tf.pow((self.logits + 1), 2)))

                if self.gradient_penalty_weight > 0.0:
                    BS, length = shape_list(self.states[0])
                    self.expert_states = tf.compat.v1.placeholder(tf.float32, [None, length], name='exp_state')
                    self.expert_acts = tf.compat.v1.placeholder(tf.int32, [None, 1], name='expert_acts')
                    self.expert_states_n = tf.compat.v1.placeholder(tf.float32, [None, length], name='exp_state_n')

                    with tf.compat.v1.variable_scope('net', reuse=True):
                        net_output = network(states=[self.expert_states], states_n=[self.expert_states_n], act=self.expert_acts,
                                                           with_action=self.with_action, actions_size=self.actions_size)

                    grad_tfs = tf.gradients(net_output[0], net_output[1:])
                    grad_tfs = [tf.reshape(grad, [BS, -1]) for grad in grad_tfs]
                    grad_tf = tf.concat(grad_tfs, axis=-1)
                    norm_tf = tf.reduce_sum(tf.square(grad_tf), axis=-1)
                    loss_tf = 0.5 * tf.reduce_mean(norm_tf)
                    self.loss += (self.gradient_penalty_weight * loss_tf)


                self.step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self):
        losses = []

        expert_traj = self.expert_traj
        policy_traj = self.policy_traj

        #self.clean_buffer()

        # Update reward model for num_itr mini-batch steps
        for it in range(self.num_itr):

            # expert_batch_idxs = np.random.randint(0, len(expert_traj['obs']), self.batch_size)
            # policy_batch_idxs = np.random.randint(0, len(policy_traj['obs']), self.batch_size)
            expert_batch_idxs = np.random.choice(len(expert_traj['obs']), self.batch_size, replace=False)
            policy_batch_idxs = np.random.choice(len(policy_traj['obs']), self.batch_size, replace=False)

            expert_obs = [expert_traj['obs'][id] for id in expert_batch_idxs]
            policy_obs = [policy_traj['obs'][id] for id in policy_batch_idxs]

            expert_obs_n = [expert_traj['obs_n'][id] for id in expert_batch_idxs]
            policy_obs_n = [policy_traj['obs_n'][id] for id in policy_batch_idxs]

            expert_acts = [expert_traj['acts'][id] for id in expert_batch_idxs]
            policy_acts = [policy_traj['acts'][id] for id in policy_batch_idxs]

            labels = np.ones((self.batch_size, 1))
            labels = np.concatenate([labels, np.zeros((self.batch_size, 1))])

            e_states = self.obs_to_state(expert_obs)
            p_states = self.obs_to_state(policy_obs)

            e_states_n = self.obs_to_state(expert_obs_n)
            p_states_n = self.obs_to_state(policy_obs_n)

            all_states = []
            all_states_n = []

            for i in range(len(e_states)):
                all_states.append(np.concatenate([e_states[i], p_states[i]], axis=0))
                all_states_n.append(np.concatenate([e_states_n[i], p_states_n[i]], axis=0))

            feed_dict = {}
            for i in range(len(self.states)):
                feed_dict[self.states[i]] = all_states[i]
                feed_dict[self.states_n[i]] = all_states_n[i]

            feed_dict[self.labels] = labels

            if self.with_action:
                all_acts = np.concatenate([expert_acts, policy_acts], axis=0)
                all_acts = np.expand_dims(all_acts, axis=1)

                feed_dict[self.act] = all_acts

            if self.gradient_penalty_weight > 0.0:
                feed_dict[self.expert_states] = e_states[0]
                feed_dict[self.expert_states_n] = e_states_n[0]
                feed_dict[self.expert_acts] = np.expand_dims(expert_acts, axis=1)

            loss, discriminator, _ = self.sess.run([self.loss, self.discriminator, self.step], feed_dict=feed_dict)

            losses.append(loss)

        return np.mean(losses), 0

    # Return the - log( 1 - D(s,a))
    def eval_discriminator(self, obs, obs_n, probs, acts=None):
        states = self.obs_to_state(obs)
        states_n = self.obs_to_state(obs_n)

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.states[i]] = states[i]
            feed_dict[self.states_n[i]] = states_n[i]

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.act] = acts

        rew, logits = self.sess.run([self.discriminator, self.logits], feed_dict=feed_dict)
        rew = np.reshape(rew, (-1))
        logits = np.reshape(logits, (-1))

        # Reward from original GAIL
        #rew = - np.log(1 - rew + eps)
        # Reward from AMP
        rew = np.maximum(0, 1 - 0.25 * np.power((logits - 1), 2))

        # Add the rewards to the normalization statistics
        if not isinstance(self.r_norm, DynamicRunningStat):
            for r in rew:
                self.r_norm.push(r)

        return rew

    def eval(self, obs, obs_n, acts=None, probs=None):

        return self.eval_discriminator(obs, obs_n, probs, acts)

    def eval_latent(self, obs, obs_n, acts=None, probs=None):
        states = self.obs_to_state(obs)
        states_n = self.obs_to_state(obs_n)

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.states[i]] = states[i]
            feed_dict[self.states_n[i]] = states_n[i]

        if self.with_action and acts is not None:
            acts = np.expand_dims(acts, axis=1)
            feed_dict[self.act] = acts

        rew = self.sess.run([self.discriminator], feed_dict=feed_dict)
        rew = np.reshape(rew, (-1))

        # Reward from original GAIL
        rew = - np.log(1 - rew + eps)
        # Reward from AMP
        # rew = np.maximum(0, 1 - 0.25 * np.power((rew - 1), 2))

        # Add the rewards to the normalization statistics
        if not isinstance(self.r_norm, DynamicRunningStat):
            for r in rew:
                self.r_norm.push(r)

        return rew
