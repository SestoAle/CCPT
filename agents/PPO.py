import tensorflow as tf
import tensorflow_probability as tfp
import random
import numpy as np
from math import sqrt
import utils
from copy import deepcopy
from layers.layers import transformer, circ_conv1d

import os

eps = 1e-5


# Actor-Critic PPO. The Actor is independent by the Critic.
class PPO:
    # PPO agent
    def __init__(self, sess, input_spec, network_spec, obs_to_state, p_lr=5e-6, v_lr=5e-4, batch_fraction=0.33,
                 p_num_itr=20, v_num_itr=20, v_batch_fraction=0.33, previous_act=False,
                 distribution='gaussian', action_type='continuous', action_size=2, action_min_value=-1,
                 action_max_value=1, frequency_mode='episodes',
                 epsilon=0.2, c1=0.5, c2=0.01, discount=0.9, lmbda=1.0, name='ppo', memory=10, norm_reward=False,
                 model_name='agent',
                 # LSTM
                 recurrent=False, recurrent_length=8, recurrent_baseline=False,

                 **kwargs):

        # Model parameters
        self.sess = sess
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_fraction = batch_fraction
        self.v_batch_fraction = v_batch_fraction
        self.p_num_itr = p_num_itr
        self.v_num_itr = v_num_itr
        self.name = name
        self.norm_reward = norm_reward
        self.model_name = model_name
        self.frequency_mode = frequency_mode
        # Functions that define input and network specifications
        self.input_spec = input_spec
        self.network_spec = network_spec
        self.obs_to_state = obs_to_state
        # Whether to use the previous actions or not.
        # Typically this is done with LSTM
        self.previous_act = previous_act

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda
        # Action hyper-parameters
        # Types permitted: 'discrete' or 'continuous'. Default: 'discrete'
        self.action_type = action_type if action_type == 'continuous' or action_type == 'discrete' else 'discrete'
        self.action_size = action_size
        # min and max values for continuous actions
        self.action_min_value = action_min_value
        self.action_max_value = action_max_value
        # Distribution type for continuous actions
        self.distrbution_type = distribution if distribution == 'gaussian' or distribution == 'beta' else 'gaussian'

        # Recurrent paramtere
        self.recurrent = recurrent
        self.recurrent_baseline = recurrent_baseline
        self.recurrent_length = recurrent_length
        self.recurrent_size = 256

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory
        # Create the network
        with tf.compat.v1.variable_scope(name) as vs:

            # Define the input placeholders form external function
            self.inputs = self.input_spec()

            # Actor network
            with tf.compat.v1.variable_scope('actor'):
                # Previous prob, for training
                self.old_logprob = tf.compat.v1.placeholder(tf.float32, [None, ], name='old_prob')
                self.baseline_values = tf.compat.v1.placeholder(tf.float32, [None, ], name='baseline_values')
                self.reward = tf.compat.v1.placeholder(tf.float32, [None, ], name='rewards')

                # Network specification from external function
                self.main_net = self.network_spec(self.inputs)

                # Final p_layers
                self.p_network = self.linear(self.main_net, 512, name='p_fc1', activation=tf.nn.relu)
                if self.previous_act:
                    self.p_network = tf.concat([self.p_network, self.inputs[-1]], axis=1)

                if not self.recurrent:
                    self.p_network = self.linear(self.p_network, 128, name='p_fc2', activation=tf.nn.relu)
                else:
                    # The last FC layer will be replaced by an LSTM layer.
                    # Recurrent network needs more variables

                    # Get batch size and number of feature of the previous layer
                    bs, feature = utils.shape_list(self.p_network)
                    self.recurrent_train_length = tf.compat.v1.placeholder(tf.int32)
                    self.sequence_lengths = tf.compat.v1.placeholder(tf.int32, [None, ])
                    self.p_network = tf.reshape(self.p_network, [bs / self.recurrent_train_length,
                                                                 self.recurrent_train_length, feature])
                    # Define the RNN cell
                    self.rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.recurrent_size,
                                                                           state_is_tuple=True, activation=tf.nn.tanh)
                    # Define state_in for the cell
                    self.state_in = self.rnn_cell.zero_state(bs, tf.float32)

                    # Apply rnn
                    self.rnn, self.rnn_state = tf.compat.v1.nn.dynamic_rnn(
                        inputs=self.p_network, cell=self.rnn_cell, dtype=tf.float32, initial_state=self.state_in,
                        sequence_length=self.sequence_lengths
                    )

                    self.rnn_batch = tf.reshape(self.rnn, [-1, self.recurrent_size])

                    # Take only the last state of the sequence
                    self.p_network = self.rnn_state.h

                # If action_space is discrete, then it is a Categorical distribution
                if self.action_type == 'discrete':
                    # Probability distribution
                    self.probs = self.linear(self.p_network, action_size, activation=tf.nn.softmax,
                                             name='probs') + eps
                    # Distribution to sample
                    self.dist = tfp.distributions.Categorical(probs=self.probs, allow_nan_stats=False)
                # If action_space is continuous, then it is a Gaussian distribution
                elif self.action_type == 'continuous':
                    # Beta distribution
                    if self.distrbution_type == 'beta':
                        self.alpha = self.linear(self.p_network, self.action_size, activation=tf.nn.softplus,
                                                 name='alpha') + 1
                        self.beta = self.linear(self.p_network, self.action_size, activation=tf.nn.softplus,
                                                name='beta') + 1
                        # This is useless, just to return something in eval() method
                        self.probs = tf.concat([self.alpha, self.beta], axis=1, name='probs')
                        self.dist = tfp.distributions.Beta(self.alpha, self.beta, allow_nan_stats=False,
                                                           name='Beta')

                    # Gaussian Distribution
                    elif self.distrbution_type == 'gaussian':
                        self.mean = self.linear(self.p_network, self.action_size, activation=None, name='mean')
                        self.variance = self.linear(self.p_network, self.action_size, activation=tf.nn.softplus,
                                                    name='var')
                        self.variance = tf.clip_by_value(self.variance, -20, 2)
                        # This is useless, just to return something in eval() method
                        self.probs = tf.concat([self.mean, self.variance], axis=1, name='probs')
                        # Normal distribution to sample
                        self.dist = tfp.distributions.Normal(self.mean, self.variance, allow_nan_stats=False,
                                                             name='Normal')

                # Sample action
                if self.action_type == 'discrete':
                    self.action = self.dist.sample(name='action')
                    self.log_prob = self.dist.log_prob(self.action)
                elif self.action_type == 'continuous':
                    self.action = self.dist.sample(name='action')
                    self.log_prob = self.dist.log_prob(self.action)

                # If there are more than 1 continuous actions, do the mean of log_probs
                if self.action_size > 1 and self.action_type == 'continuous':
                    self.log_prob = tf.reduce_sum(self.log_prob, axis=1)

                # If continuous, bound the actions between min_action and max_action
                if self.action_type == 'continuous':
                    # Beta Distribution
                    if self.distrbution_type == 'beta':
                        self.action = self.action_min_value + (
                                self.action_max_value - self.action_min_value) * self.action
                    # Gaussian Distribution
                    elif self.distrbution_type == 'gaussian':
                        self.action = tf.tanh(self.action)
                        one = tf.constant(value=1.0, dtype=tf.float32)
                        half = tf.constant(value=0.5, dtype=tf.float32)
                        min_value = tf.constant(value=self.action_min_value, dtype=tf.float32)
                        max_value = tf.constant(value=self.action_max_value, dtype=tf.float32)
                        self.action = min_value + (max_value - min_value) * half * (self.action + one)

                # Get probability of a given action - useful for training
                with tf.compat.v1.variable_scope('eval_with_action'):
                    if self.action_type == 'discrete':
                        self.eval_action = tf.compat.v1.placeholder(tf.int32, [None, ], name='eval_action')
                        self.real_action = self.eval_action
                    elif self.action_type == 'continuous':
                        self.eval_action = tf.compat.v1.placeholder(tf.float32, [None, self.action_size],
                                                                    name='eval_action')

                        # Inverse normalization actions between min_value and max_value
                        # Beta Distribution
                        if self.distrbution_type == 'beta':
                            self.real_action = (self.eval_action - self.action_min_value) / (
                                    self.action_max_value - self.action_min_value)

                        # Gaussian Distribution
                        elif self.distrbution_type == 'gaussian':
                            # Tanh inverse transformation
                            one = tf.constant(value=1.0, dtype=tf.float32)
                            two = tf.constant(value=2.0, dtype=tf.float32)
                            min_value = tf.constant(value=self.action_min_value, dtype=tf.float32)
                            max_value = tf.constant(value=self.action_max_value, dtype=tf.float32)
                            self.real_action = two * (self.eval_action - min_value) / (max_value - min_value) - one
                            self.real_action = tf.clip_by_value(t=self.real_action, clip_value_min=-1+eps, clip_value_max=1-eps)
                            self.real_action = tf.atanh(self.real_action)

                    self.log_prob_with_action = self.dist.log_prob(self.real_action)
                    # If there are more than 1 continuous actions, do the mean of log_probs
                    if self.action_size > 1 and self.action_type == 'continuous':
                        self.log_prob_with_action = tf.reduce_sum(self.log_prob_with_action, axis=1)

            # Critic network
            with tf.compat.v1.variable_scope('critic'):

                # V Network specification
                self.v_network = self.network_spec(self.inputs)

                # Final p_layers
                if not self.recurrent_baseline:
                    self.v_network = self.linear(self.v_network, 512, name='v_fc1', activation=tf.nn.relu)
                else:
                    # The last FC layer will be replaced by an LSTM layer.
                    # Recurrent network needs more variables

                    # Get batch size and number of feature of the previous layer
                    bs, feature = utils.shape_list(self.v_network)
                    self.v_network = tf.reshape(self.v_network, [bs / self.recurrent_train_length,
                                                                 self.recurrent_train_length, feature])
                    # Define the RNN cell
                    self.v_rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.recurrent_size,
                                                                             state_is_tuple=True, activation=tf.nn.tanh)
                    # Define state_in for the cell
                    self.v_state_in = self.v_rnn_cell.zero_state(bs, tf.float32)

                    # Apply rnn
                    self.v_rnn, self.v_rnn_state = tf.compat.v1.nn.dynamic_rnn(
                        inputs=self.v_network, cell=self.v_rnn_cell, dtype=tf.float32, initial_state=self.v_state_in,
                        sequence_length=self.sequence_lengths
                    )

                    # Take only the last state of the sequence
                    self.v_network = self.v_rnn_state.h

                self.v_network = self.linear(self.v_network, 128, name='v_fc2', activation=tf.nn.relu)

                # Value function
                self.value = tf.squeeze(self.linear(self.v_network, 1))

            # Advantage
            # Advantage (reward - baseline)
            self.advantage = self.reward - self.baseline_values

            # L_clip loss
            self.ratio = tf.exp(self.log_prob_with_action - self.old_logprob)
            self.surr1 = self.ratio * self.advantage
            self.surr2 = tf.clip_by_value(self.ratio, 1 - self.epsilon, 1 + self.epsilon) * self.advantage
            self.clip_loss = tf.minimum(self.surr1, self.surr2)

            # Value function loss
            self.mse_loss = tf.compat.v1.losses.mean_squared_error(self.reward, self.value)

            # Entropy bonus
            self.entr_loss = self.dist.entropy()
            # If there are more than 1 continuous actions, do the mean of entropies
            if self.action_size > 1 and self.action_type == 'continuous':
                self.entr_loss = tf.reduce_sum(self.entr_loss, axis=1)

            # Total loss
            self.total_loss = - tf.reduce_mean(self.clip_loss + self.c2 * (self.entr_loss + eps))

            # Policy Optimizer
            self.p_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.p_lr).minimize(self.total_loss)
            # Value Optimizer
            self.v_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.v_lr).minimize(self.mse_loss)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    ## Layers
    def linear(self, inp, inner_size, name='linear', bias=True, activation=None, init=None):
        with tf.compat.v1.variable_scope(name):
            lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                            kernel_initializer=init)
            return lin

    def conv_layer_2d(self, input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv',
                      activation=None, bias=True):

        with tf.compat.v1.variable_scope(name):
            conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                              activation=activation, use_bias=bias)
            return conv

    def embedding(self, input, indices, size, name='embs'):
        with tf.compat.v1.variable_scope(name):
            shape = (indices, size)
            stddev = min(0.1, sqrt(2.0 / (utils.product(xs=shape[:-1]) + shape[-1])))
            initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
            W = tf.Variable(
                initial_value=initializer, trainable=True, validate_shape=True, name='W',
                dtype=tf.float32, shape=shape
            )
            return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

    def sample_batch_for_recurrent(self, length, batch_size):
        all_idxs = np.arange(len(self.buffer['states']))
        new_ep_lengths = deepcopy(self.buffer['episode_lengths'])

        for i, ep in enumerate(self.buffer['episode_lengths']):

            if ep < length:
                index = np.cumsum(new_ep_lengths)[i] - 1
                added_length = length - ep
                all_idxs = np.insert(all_idxs, index, np.ones(added_length) * all_idxs[index])
                new_ep_lengths[i] = length
            else:
                new_ep_lengths[i] = ep

        max_seq_steps = np.cumsum(new_ep_lengths) - length + 1
        min_seq_steps = np.concatenate([[0], np.cumsum(new_ep_lengths)])
        val_idxs = np.concatenate([np.arange(min, max) for min, max in zip(min_seq_steps, max_seq_steps)])

        batch_size = np.minimum(len(val_idxs), batch_size)

        val_idxs_first_step = np.random.choice(val_idxs, batch_size, replace=False)
        minibatch_idxs = []
        minibatch_idxs_last_step = []
        sequence_lengths = []
        minibatch_idxs_first_step = []
        for first in val_idxs_first_step:
            minibatch_idxs.extend(all_idxs[first:first + length])
            minibatch_idxs_last_step.append(all_idxs[first + length - 1])
            minibatch_idxs_first_step.append(all_idxs[first])
            parent_ep = np.sum(np.cumsum(self.buffer['episode_lengths']) <= all_idxs[first])

            sequence_lengths.append(np.minimum(length, self.buffer['episode_lengths'][parent_ep]))

        return minibatch_idxs, minibatch_idxs_last_step, minibatch_idxs_first_step, sequence_lengths

    # Train loop
    def train(self):
        losses = []
        v_losses = []

        # Get batch size based on batch_fraction
        batch_size = int(len(self.buffer['states']) * self.v_batch_fraction)
        if self.recurrent_baseline:
            batch_size = int(len(self.buffer['states']) * self.v_batch_fraction)

        # Before training, compute discounted reward
        discounted_rewards = self.compute_discounted_reward()

        # Train the value function
        for it in range(self.v_num_itr):
            if not self.recurrent_baseline:
                # Take a mini-batch of batch_size experience
                mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            else:
                # Take the idxs of the sequences AND the idx of the last state of the sequence
                mini_batch_idxs, mini_batch_idxs_last_step, mini_batch_idxs_first_step, sequence_lengths = \
                    self.sample_batch_for_recurrent(self.recurrent_length, batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
                v_internal_states_c = [self.buffer['v_internal_states_c'][id] for id in mini_batch_idxs_first_step]
                v_internal_states_h = [self.buffer['v_internal_states_h'][id] for id in mini_batch_idxs_first_step]
                tmp_batch_size = len(states_mini_batch) // self.recurrent_length
                v_internal_states_c = np.reshape(v_internal_states_c, [tmp_batch_size, -1])
                v_internal_states_h = np.reshape(v_internal_states_h, [tmp_batch_size, -1])
                v_internal_states = (v_internal_states_c, v_internal_states_h)
                mini_batch_idxs = mini_batch_idxs_last_step

            try:
                rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            except Exception as e:
                print(e)
                print(mini_batch_idxs)
                input('....')
            # Reshape problem, why?
            rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])

            # Get DeepCrawl state
            # Convert the observation to states
            states = self.obs_to_state(states_mini_batch)
            feed_dict = self.create_state_feed_dict(states)

            # Update feed dict for training
            feed_dict[self.reward] = rewards_mini_batch

            if not self.recurrent_baseline:
                v_loss, step = self.sess.run([self.mse_loss, self.v_step], feed_dict=feed_dict)
            else:
                # If recurrent, we need to pass the internal state and the recurrent_length
                feed_dict[self.v_state_in] = v_internal_states
                feed_dict[self.sequence_lengths] = sequence_lengths
                feed_dict[self.recurrent_train_length] = self.recurrent_length
                v_loss, step = self.sess.run([self.mse_loss, self.v_step], feed_dict=feed_dict)

            v_losses.append(v_loss)

        # Compute GAE for rewards. If lambda == 1, they are discounted rewards
        # Compute values for each state

        num_batches = 10
        batch_size = int(np.ceil(len(self.buffer['states'])/num_batches))
        v_values = []
        for i in range(num_batches):
            states = self.obs_to_state(self.buffer['states'][batch_size*i:batch_size*i + batch_size])
            feed_dict = self.create_state_feed_dict(states)
            if self.recurrent_baseline:
                v_internal_states_c = self.buffer['v_internal_states_c'][batch_size*i:batch_size*i + batch_size]
                v_internal_states_h = self.buffer['v_internal_states_h'][batch_size*i:batch_size*i + batch_size]
                v_internal_states_c = np.reshape(v_internal_states_c, [batch_size, -1])
                v_internal_states_h = np.reshape(v_internal_states_h, [batch_size, -1])
                v_internal_states = (v_internal_states_c, v_internal_states_h)
                feed_dict[self.v_state_in] = v_internal_states
                feed_dict[self.sequence_lengths] = np.ones(batch_size)
                feed_dict[self.recurrent_train_length] = 1

            v_values.extend(self.sess.run(self.value, feed_dict=feed_dict))

        v_values = np.append(v_values, 0)

        discounted_rewards = self.compute_gae(v_values)

        # Get batch size based on batch_fraction
        batch_size = int(len(self.buffer['states']) * self.batch_fraction)
        if self.recurrent:
            batch_size = int(len(self.buffer['states']) * self.batch_fraction)

        # Train the policy
        for it in range(self.p_num_itr):

            if not self.recurrent:
                # Take a mini-batch of batch_size experience
                mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            else:
                # Take the idxs of the sequences AND the idx of the last state of the sequence
                mini_batch_idxs, mini_batch_idxs_last_step, mini_batch_idxs_first_step, sequence_lengths = \
                    self.sample_batch_for_recurrent(self.recurrent_length, batch_size)
                states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
                internal_states_c = [self.buffer['internal_states_c'][id] for id in mini_batch_idxs_first_step]
                internal_states_h = [self.buffer['internal_states_h'][id] for id in mini_batch_idxs_first_step]
                tmp_batch_size = len(states_mini_batch) // self.recurrent_length
                internal_states_c = np.reshape(internal_states_c, [tmp_batch_size, -1])
                internal_states_h = np.reshape(internal_states_h, [tmp_batch_size, -1])
                internal_states = (internal_states_c, internal_states_h)
                mini_batch_idxs = mini_batch_idxs_last_step

            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]
            old_probs_mini_batch = [self.buffer['old_probs'][id] for id in mini_batch_idxs]
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]

            # Get DeepCrawl state
            # Convert the observation to states
            states = self.obs_to_state(states_mini_batch)
            feed_dict = self.create_state_feed_dict(states)

            # Get the baseline values
            v_values_mini_batch = [v_values[id] for id in mini_batch_idxs]

            # Reshape problem, why?
            rewards_mini_batch = np.reshape(rewards_mini_batch, [-1, ])
            old_probs_mini_batch = np.reshape(old_probs_mini_batch, [-1, ])
            v_values_mini_batch = np.reshape(v_values_mini_batch, [-1, ])

            # Update feed dict for training
            feed_dict[self.reward] = rewards_mini_batch
            feed_dict[self.old_logprob] = old_probs_mini_batch
            feed_dict[self.eval_action] = actions_mini_batch
            feed_dict[self.baseline_values] = v_values_mini_batch

            if not self.recurrent:
                loss, step = self.sess.run([self.total_loss, self.p_step], feed_dict=feed_dict)
            else:
                # If recurrent, we need to pass the internal state and the recurrent_length
                feed_dict[self.state_in] = internal_states
                feed_dict[self.sequence_lengths] = sequence_lengths
                feed_dict[self.recurrent_train_length] = self.recurrent_length
                loss, step = self.sess.run([self.total_loss, self.p_step], feed_dict=feed_dict)
            losses.append(loss)

        return np.mean(losses)

    # Eval sampling the action (done by the net)
    def eval(self, state):
        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        action, logprob, probs = self.sess.run([self.action, self.log_prob, self.probs], feed_dict=feed_dict)

        return action, logprob, probs

    # Eval sampling the action, but with recurrent: it needs the internal hidden state
    def eval_recurrent(self, state, internal, v_internal=None):
        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        # Pass the internal state
        feed_dict[self.state_in] = internal
        feed_dict[self.recurrent_train_length] = 1
        feed_dict[self.sequence_lengths] = [1]
        action, logprob, probs, internal = self.sess.run([self.action, self.log_prob, self.probs, self.rnn_state],
                                                         feed_dict=feed_dict)

        if self.recurrent_baseline:
            feed_dict[self.v_state_in] = v_internal
            v_internal = self.sess.run([self.v_state_in], feed_dict=feed_dict)

        # Return is equal to eval(), but with the new internal state
        return action, logprob, probs, internal, v_internal

    # Eval with argmax, but with recurrent: it needs the internal hidden state
    def eval_recurrent_max(self, state, internal, v_internal=None):
        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)

        # Pass the internal state
        feed_dict[self.state_in] = internal
        feed_dict[self.recurrent_train_length] = 1
        feed_dict[self.sequence_lengths] = [1]
        action, logprob, probs, internal = self.sess.run([self.action, self.log_prob, self.probs, self.rnn_state],
                                                         feed_dict=feed_dict)
        if self.recurrent_baseline:
            feed_dict[self.v_state_in] = v_internal
            v_internal = self.sess.run([self.v_state_in], feed_dict=feed_dict)

        # Return is equal to eval(), but with the new internal state
        return [np.argmax(probs)], logprob, probs, internal, v_internal

    # Eval with argmax
    def eval_max(self, state):

        state = self.obs_to_state(state)
        feed_dict = self.create_state_feed_dict(state)


        if self.action_type == 'continuous':
            if self.distrbution_type == 'beta':
                # Return mean as deterministic action
                beta, alpha = self.sess.run([self.beta, self.alpha], feed_dict=feed_dict)
                action = beta / (alpha + beta)
                action = self.action_min_value + (
                                self.action_max_value - self.action_min_value) * action
            else:
                # Return mean as deterministic action
                action = self.sess.run([self.mean], feed_dict=feed_dict)
                action = tf.tanh(action)
                one = tf.constant(value=1.0, dtype=tf.float32)
                half = tf.constant(value=0.5, dtype=tf.float32)
                min_value = tf.constant(value=self.action_min_value, dtype=tf.float32)
                max_value = tf.constant(value=self.action_max_value, dtype=tf.float32)
                action = min_value + (max_value - min_value) * half * (action + one)

        else:
            probs = self.sess.run([self.probs], feed_dict=feed_dict)
            action = np.argmax(probs)

        return action

    # Eval with a given action
    def eval_action(self, states, actions):

        state = self.obs_to_state(states)
        feed_dict = self.create_state_feed_dict(state)
        feed_dict[self.eval_action] = actions

        logprobs = self.sess.run([self.log_prob_with_action], feed_dict=feed_dict)[0]

        logprobs = np.reshape(logprobs, [-1, 1])

        return logprobs

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.inputs[i]] = states[i]

        return feed_dict

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['episode_lengths'] = []
        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []
        if self.recurrent:
            self.buffer['internal_states_c'] = []
            self.buffer['internal_states_h'] = []
            self.buffer['v_internal_states_c'] = []
            self.buffer['v_internal_states_h'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals,
                      internal_states_c=None, internal_states_h=None,
                      v_internal_states_c=None, v_internal_states_h=None):

        # If we store more than memory episodes, remove the last episode
        if self.frequency_mode == 'episodes':
            if len(self.buffer['episode_lengths']) + 1 >= self.memory + 1:
                idxs_to_remove = self.buffer['episode_lengths'][0]
                del self.buffer['states'][:idxs_to_remove]
                del self.buffer['actions'][:idxs_to_remove]
                del self.buffer['old_probs'][:idxs_to_remove]
                del self.buffer['states_n'][:idxs_to_remove]
                del self.buffer['rewards'][:idxs_to_remove]
                del self.buffer['terminals'][:idxs_to_remove]
                del self.buffer['episode_lengths'][0]
                if self.recurrent:
                    del self.buffer['internal_states_c'][:idxs_to_remove]
                    del self.buffer['internal_states_h'][:idxs_to_remove]
                if self.recurrent_baseline:
                    del self.buffer['v_internal_states_c'][:idxs_to_remove]
                    del self.buffer['v_internal_states_h'][:idxs_to_remove]
        # If we store more than memory timesteps, remove the last timestep
        elif self.frequency_mode == 'timesteps':
            if (len(self.buffer['states']) + 1 > self.memory):
                del self.buffer['states'][0]
                del self.buffer['actions'][0]
                del self.buffer['old_probs'][0]
                del self.buffer['states_n'][0]
                del self.buffer['rewards'][0]
                del self.buffer['terminals'][0]
                if self.recurrent:
                    del self.buffer['internal_states_c'][0]
                    del self.buffer['internal_states_h'][0]
                if self.recurrent_baseline:
                    del self.buffer['v_internal_states_c'][0]
                    del self.buffer['v_internal_states_h'][0]

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)
        if self.recurrent:
            self.buffer['internal_states_c'].append(internal_states_c)
            self.buffer['internal_states_h'].append(internal_states_h)
        if self.recurrent_baseline:
            self.buffer['v_internal_states_c'].append(v_internal_states_c)
            self.buffer['v_internal_states_h'].append(v_internal_states_h)
        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if self.frequency_mode=='episodes':
            if terminals == 1 or terminals == 2:
                self.buffer['episode_lengths'].append(
                    int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))
        else:
            self.buffer['episode_lengths'] = []
            for i, t in enumerate(self.buffer['terminals']):
                if t == 1 or t == 2:
                    self.buffer['episode_lengths'].append(
                        int(i + 1 - np.sum(self.buffer['episode_lengths'])))

    # Change rewards in buffer to discounted rewards
    def compute_discounted_reward(self):

        discounted_rewards = []
        discounted_reward = 0
        # The discounted reward can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            if terminal == 1:
                discounted_reward = 0
                # state = self.obs_to_state([self.buffer['states_n'][i]])
                # feed_dict = self.create_state_feed_dict(state)
                # discounted_reward = self.sess.run([self.value], feed_dict)[0]
            elif terminal == 2:
                state = self.obs_to_state([self.buffer['states_n'][i]])
                feed_dict = self.create_state_feed_dict(state)
                discounted_reward = self.sess.run([self.value], feed_dict)[0]

            discounted_reward = reward + (self.discount * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing reward
        if self.norm_reward:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                        np.std(discounted_rewards) + eps)

        return discounted_rewards

    # Change rewards in buffer to discounted rewards or GAE rewards (if lambda == 1, gae == discounted)
    def compute_gae(self, v_values):

        rewards = []
        gae = 0

        # The gae rewards can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            m = 1
            if terminal == 1:
                m = 0
                gae = 0

            delta = reward + self.discount * v_values[i + 1] * m - v_values[i]
            gae = delta + self.discount * self.lmbda * m * gae
            discounted_reward = gae + v_values[i]

            rewards.insert(0, discounted_reward)

        # Normalizing
        if self.norm_reward:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        tf.compat.v1.disable_eager_execution()
        self.saver.save(self.sess, '{}/{}'.format(folder, name))

        if True:
            graph_def = self.sess.graph.as_graph_def()

            # freeze_graph clear_devices option
            for node in graph_def.node:
                node.device = ''

            graph_def = tf.compat.v1.graph_util.remove_training_nodes(input_graph=graph_def)
            output_node_names = [
                'ppo/actor/add',
                'ppo/actor/ppo_actor_Categorical/action/Reshape_2',
                'ppo/critic/Squeeze'
            ]

            # implies tf.compat.v1.graph_util.extract_sub_graph
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=self.sess, input_graph_def=graph_def,
                output_node_names=output_node_names
            )
            graph_path = tf.io.write_graph(
                graph_or_graph_def=graph_def, logdir=folder,
                name=(name + '.pb'), as_text=False
            )

        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        # self.saver = tf.compat.v1.train.import_meta_graph('{}/{}.meta'.format(folder, name))
        tf.compat.v1.disable_eager_execution()
        self.saver.restore(self.sess, '{}/{}'.format(folder, name))

        print('Model loaded correctly!')
        return
