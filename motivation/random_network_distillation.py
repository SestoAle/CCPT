import tensorflow as tf
import numpy as np
from layers.layers import *

from utils import DynamicRunningStat, LimitedRunningStat, RunningStat
import random


eps = 1e-12

class RND:
    # Random Network Distillation class
    def __init__(self, sess, input_spec, network_spec_target, network_spec_predictor, obs_to_state, lr=7e-5,
                 buffer_size=1e5, batch_size=128, num_epochs=3,
                 motivation_weight=1., obs_normalization=False,
                 num_itr=3, name='rnd', **kwargs):

        # Used to normalize the intrinsic reward due to arbitrary scale
        self.r_norm = RunningStat()
        self.obs_norm = RunningStat(shape=(9269))
        self.obs_normalization = obs_normalization

        # The tensorflow session
        self.sess = sess

        # Model hyperparameters
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_itr = num_itr
        self.num_epochs = num_epochs
        # Functions that define input and network specifications
        self.input_spec = input_spec
        self.network_spec_target = network_spec_target
        self.network_spec_predictor = network_spec_predictor
        self.obs_to_state = obs_to_state
        # Weight of the reward output by the motivation model
        self.motivation_weight = motivation_weight

        # Buffer of experience
        self.buffer = []

        with tf.compat.v1.variable_scope(name) as vs:
            # Input placeholders, they depend on DeepCrawl
            self.inputs = self.input_spec()

            # Target network, it must remain fixed during all the training
            with tf.compat.v1.variable_scope('target'):
                # Network specification from external function
                self.target = self.network_spec_target(self.inputs)

            # Predictor network
            with tf.compat.v1.variable_scope('predictor'):
                # Network specification from external function
                self.predictor = self.network_spec_predictor(self.inputs)

            # For fixed target labels, use a placeholder in order to NOT update the target network
            _, latent = shape_list(self.target)
            self.target_labels = tf.compat.v1.placeholder(tf.float32, [None, latent], name='target_labels')

            self.reward_loss = tf.compat.v1.losses.mean_squared_error(self.target_labels, self.predictor)
            #self.rewards = tf.math.squared_difference(self.target_labels, self.predictor)
            self.rewards = tf.reduce_sum(tf.math.pow(self.target_labels - self.predictor, 2), axis=1)

            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.reward_loss))
            gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 1.0)
            self.step = optimizer.apply_gradients(zip(gradients, variables))

        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

    # Fit function
    def train(self):
        losses = []

        # If we want to use observation normalization, normalize the buffer
        if self.obs_normalization:
            self.normalize_buffer()

        for it in range(self.num_itr):
        # for e in range(self.num_epochs):
        #     num_batches = int(np.ceil(len(self.buffer)/self.batch_size))
        #     all_index = np.arange(len(self.buffer))
        #     np.random.shuffle(all_index)
            # for b in range(num_batches):
                # Take a mini-batch of batch_size experience
                mini_batch_idxs = np.random.choice(len(self.buffer), self.batch_size, replace=False)
                # mini_batch_idxs = all_index[i*self.batch_size: i*self.batch_size + self.batch_size]

                mini_batch = [self.buffer[id] for id in mini_batch_idxs]

                # Convert the observation to states
                states = self.obs_to_state(mini_batch)

                # Create the feed dict for the target network
                feed_target = self.create_state_feed_dict(states)

                # Get the target prediction (without training it)
                target_labels = self.sess.run([self.target], feed_target)[0]

                # Get the predictor estimation
                feed_predictor = self.create_state_feed_dict(states)
                feed_predictor[self.target_labels] = target_labels

                # Update the predictor networks
                loss, step, rews = self.sess.run([self.reward_loss, self.step, self.rewards], feed_predictor)

                losses.append(loss)

        # Update Dynamic Running Stat
        if isinstance(self.r_norm, DynamicRunningStat):
            self.r_norm.reset()

        self.buffer = []

        # Return the mean losses of all the iterations
        return np.mean(losses)

    # Eval function
    def eval(self, obs):

        # Normalize observation
        if self.obs_normalization:
            self.normalize_states(obs)

        # Convert the observation to states
        states = self.obs_to_state(obs)

        # Create the feed dict for the target network
        feed_target = self.create_state_feed_dict(states)

        # Get the target prediction (without training it)
        target_labels = self.sess.run([self.target], feed_target)[0]

        # Get the predictor estimation
        feed_predictor = self.create_state_feed_dict(states)
        feed_predictor[self.target_labels] = target_labels

        # Compute the MSE to use as reward (after normalization)
        # Update the predictor networks
        rewards = self.sess.run(self.rewards, feed_predictor)
        rewards = np.reshape(rewards, (-1))

        # Add the rewards to the normalization statistics
        if not isinstance(self.r_norm, DynamicRunningStat):
            for r in rewards:
                self.r_norm.push(r)

        return rewards

    # Eval function
    def eval_latent(self, obs):

        # Normalize observation
        if self.obs_normalization:
            self.normalize_states(obs)

        # Convert the observation to states
        states = self.obs_to_state(obs)

        # Create the feed dict for the target network
        feed_target = self.create_state_feed_dict(states)

        # Get the target prediction (without training it)
        target_labels = self.sess.run([self.target], feed_target)[0]

        # Get the predictor estimation
        feed_predictor = self.create_state_feed_dict(states)
        feed_predictor[self.target_labels] = target_labels

        # Compute the MSE to use as reward (after normalization)
        # Update the predictor networks
        rewards, latents = self.sess.run([self.rewards, self.predictor], feed_predictor)
        rewards = np.reshape(rewards, (-1))

        # Add the rewards to the normalization statistics
        if not isinstance(self.r_norm, DynamicRunningStat):
            for r in rewards:
                self.r_norm.push(r)

        return rewards, latents

    # Create a state feed_dict from states
    def create_state_feed_dict(self, states):

        feed_dict = {}
        for i in range(len(states)):
            feed_dict[self.inputs[i]] = states[i]

        return feed_dict

    # Add observation to buffer
    def add_to_buffer(self, obs, mode='random'):

        if len(self.buffer) >= self.buffer_size:
            if mode == 'random':
                index = np.random.randint(0, len(self.buffer))

                del self.buffer[index]
            else:
                del self.buffer[0]

        if self.obs_normalization:
            self.obs_norm.push(obs['global_in'])

        self.buffer.append(obs)

    # Save the entire model
    def save_model(self, name=None, folder='saved'):
        tf.compat.v1.disable_eager_execution()
        self.saver.save(self.sess, '{}/{}_rnd'.format(folder, name))

        return

    # Load entire model
    def load_model(self, name=None, folder='saved'):
        # self.saver = tf.compat.v1.train.import_meta_graph('{}/{}.meta'.format(folder, name))
        tf.compat.v1.disable_eager_execution()
        self.saver.restore(self.sess, '{}/{}_rnd'.format(folder, name))

        print('RND loaded correctly!')
        return

    # Normalize the buffer state based on the running mean
    def normalize_buffer(self):
        for state in self.buffer:
            state['global_in'] = (state['global_in'] - self.obs_norm.mean) / (self.obs_norm.std + eps)
            state['global_in'] = np.clip(state['global_in'], -5, 5)

    # Normalize input states based on the running mean
    def normalize_states(self, states):
        for state in states:
            state['global_in'] = (state['global_in'] - self.obs_norm.mean) / (self.obs_norm.std + eps)
            state['global_in'] = np.clip(state['global_in'], -5, 5)

    # Clear experience buffer
    def clear_buffer(self):
        self.buffer = []
