import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 135
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):

    global_state = states[0]
    threedgrid, rotation, agent_plane_x, agent_plane_y, agent_plane_z, goal_weight = tf.split(global_state, [125, 4, 1, 1, 1, 3], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 1000
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 1000
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_plane_y = ((agent_plane_y + 1) / 2) * 100
    agent_plane_y = tf.cast(agent_plane_y, tf.int32)
    agent_plane_y = positional_encoding(agent_plane_y, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_plane_y], axis=1)
    agent = tf.reshape(agent, (-1, 3 * 32))
    agent = linear(agent, 1024, name='global_embs', activation=tf.nn.relu)
    rotation = linear(rotation, 1024, name='grounded_embs', activation=tf.nn.relu)
    agent = tf.concat([agent, rotation], axis=1)

    threedgrid = tf.cast(tf.reshape(threedgrid, [-1, 5, 5, 5]), tf.int32)
    threedgrid = embedding(threedgrid, indices=10, size=32, name='global_embs')
    threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_01', activation=tf.nn.relu)
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_02', activation=tf.nn.relu)
    threedgrid = conv_layer_3d(threedgrid, 128, [3, 3, 3], strides=(2, 2, 2), name='conv_03', activation=tf.nn.relu)
    threedgrid = tf.reshape(threedgrid, [-1, 128])

    #goal_weight = linear(goal_weight, 1024, name='goal_embs', activation=tf.nn.relu)

    global_state = tf.concat([agent, threedgrid], axis=1)

    global_state = linear(global_state, 1024, name='out', activation=tf.nn.leaky_relu)

    return global_state

def input_spec_rnd():
    input_length = 135
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state_rnd(obs):
    global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_rnd_predictor(states):
    global_state = states[0]
    threedgrid, rotation, agent_plane_x, agent_plane_y, agent_plane_z, goal_weight = tf.split(global_state, [125, 4, 1, 1, 1, 3], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 1000
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 1000
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_plane_y = ((agent_plane_y + 1) / 2) * 100
    agent_plane_y = tf.cast(agent_plane_y, tf.int32)
    agent_plane_y = positional_encoding(agent_plane_y, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_plane_y], axis=1)
    global_state = tf.reshape(agent, (-1, 3 * 32))

    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu)
    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu)
    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu)
    global_state = linear(global_state, 128, name='latent_4', activation=tf.nn.relu)

    global_state = linear(global_state, 64, name='out')

    return global_state

def network_spec_rnd_target(states):
    global_state = states[0]
    agent_plane_x, agent_plane_y, agent_plane_z, rotation, semantic_map, goal_weight = tf.split(global_state, [1, 1, 1, 4, 125, 3], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 1000
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 1000
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_plane_y = ((agent_plane_y + 1) / 2) * 100
    agent_plane_y = tf.cast(agent_plane_y, tf.int32)
    agent_plane_y = positional_encoding(agent_plane_y, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_plane_y], axis=1)
    global_state = tf.reshape(agent, (-1, 3 * 32))

    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu)
    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu)
    global_state = linear(global_state, 64, name='out')

    return global_state

def input_spec_irl():
    input_length = 132
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    global_state_n = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    return [[global_state], act, [global_state_n]]

def obs_to_state_irl(obs):
    if len(obs[0]['global_in']) > 132:
        global_batch = np.stack([state['global_in'][:-3] for state in obs])
    else:
        global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_irl(states, states_n, act, with_action, actions_size):

    agent_state = states[0]

    agent_state = linear(agent_state, 1024, name='global_embs', activation=tf.nn.relu)
    action_state = embedding(action_state, indices=10, size=512, name='action_embs')
    action_state = tf.reshape(action_state, [-1, 512])
    action = action_state

    encoded = tf.concat([agent_state,  action], axis=1)

    global_state = linear(encoded, 1024, name='latent_1', activation=tf.nn.relu,)

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.relu,)

    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu,)


    global_state = linear(global_state, 1, name='out', 
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None, dtype=tf.dtypes.float32))

    return global_state, agent_state, action_state
