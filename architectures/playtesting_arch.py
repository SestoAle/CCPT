import tensorflow as tf
from layers.layers import *

def input_spec():
    input_length = 9269
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state(obs):
    global_batch = np.stack([np.asarray(state['global_in']) for state in obs])
    return [global_batch]

def network_spec(states):
    input_length = 71
    with_circular = False

    global_state = states[0]

    # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, threedgrid, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1, 9261, 3], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 500
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 500
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_jump = ((agent_jump + 1) / 2) * 60
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)
    agent = tf.reshape(agent, (-1, 3 * 32))
    # agent = tf.concat([agent, is_grounded, can_double_jump], axis=1)
    agent = linear(agent, 1024, name='global_embs', activation=tf.nn.relu)
    is_grounded = linear(is_grounded, 1024, name='grounded_embs', activation=tf.nn.relu)
    can_double_jump = linear(can_double_jump, 1024, name='double_embs', activation=tf.nn.relu)
    agent = tf.concat([agent, is_grounded, can_double_jump], axis=1)

    goal_weight = linear(goal_weight, 1024, name='goal_embs', activation=tf.nn.relu)

    threedgrid = tf.cast(tf.reshape(threedgrid, [-1, 21, 21, 21]), tf.int32)
    # threedgrid = tf.reshape(threedgrid, [-1, 15, 15, 15, 1])
    threedgrid = embedding(threedgrid, indices=4, size=32, name='global_embs')
    threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_01', activation=tf.nn.relu)
    #threedgrid = tf.nn.max_pool3d(threedgrid, [2, 2, 2], strides=(2, 2, 2), padding="VALID")
    threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_02', activation=tf.nn.relu)
    #threedgrid = tf.nn.max_pool3d(threedgrid, [2, 2, 2], strides=(2, 2, 2), padding="VALID")
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_03', activation=tf.nn.relu)
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_04', activation=tf.nn.relu)

    threedgrid = tf.reshape(threedgrid, [-1, 2 * 2 * 2 * 64])
    # threedgrid = linear(threedgrid, 1024, name='three_embs', activation=tf.nn.tanh)


    # target_distances = linear(target_distances, 1024, name='target_distances_emb', activation=tf.nn.tanh)

    # inventory = linear(inventory, 1024, name='inventory_embs', activation=tf.nn.tanh)

    global_state = tf.concat([agent, threedgrid, goal_weight], axis=1)

    global_state = linear(global_state, 1024, name='embs', activation=tf.nn.relu)

    return global_state

def input_spec_rnd():
    input_length = 9269
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    return [global_state]

def obs_to_state_rnd(obs):
    global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_rnd_predictor(states):
    with_circular = False

    global_state = states[0]

    # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, threedgrid, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1, 9261, 3], axis=1)
    agent_plane_x = ((agent_plane_x + 1) / 2) * 500
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 500
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_jump = ((agent_jump + 1) / 2) * 60
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)

    global_state = agent
    global_state = tf.reshape(global_state, (-1, 3 * 32))

    #goal_weight = linear(goal_weight, 1024, name='goal_embs', activation=tf.nn.relu)
    #global_state = tf.concat([global_state, goal_weight], axis=1)

    #global_state = linear(global_state, 1024, name='global_embs', activation=tf.nn.leaky_relu)


    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu,

                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu,

                          )

    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 128, name='latent_4', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 64, name='out',
                          )


    return global_state

def network_spec_rnd_target(states):
    with_circular = False

    global_state = states[0]

    # agent, goal, rays, obs = tf.split(global_state, [4, 3, 12, 21], axis=1)
    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, threedgrid, goal_weight = \
        tf.split(global_state, [1, 1, 1, 1, 1, 9261, 3], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 500
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)
    agent_plane_x = positional_encoding(agent_plane_x, 32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 500
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)
    agent_plane_z = positional_encoding(agent_plane_z, 32)

    agent_jump = ((agent_jump + 1) / 2) * 60
    agent_jump = tf.cast(agent_jump, tf.int32)
    agent_jump = positional_encoding(agent_jump, 32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)

    global_state = agent
    global_state = tf.reshape(global_state, (-1, 3 * 32))
    #goal_weight = linear(goal_weight, 1024, name='goal_embs', activation=tf.nn.relu)
    #global_state = tf.concat([global_state, goal_weight], axis=1)
    #global_state = linear(global_state, 1024, name='global_embs', activation=tf.nn.leaky_relu)


    global_state = linear(global_state, 1024, name='latent_1', activation=tf.nn.leaky_relu,
                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.leaky_relu,
                          )

    global_state = linear(global_state, 64, name='out',
                          )

    return global_state

def input_spec_irl():
    input_length = 9266
    global_state = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state')

    global_state_n = tf.compat.v1.placeholder(tf.float32, [None, input_length], name='state_n')

    act = tf.compat.v1.placeholder(tf.int32, [None, 1], name='act')

    return [[global_state], act, [global_state_n]]

def obs_to_state_irl(obs):
    if len(obs[0]['global_in']) > 9266:
        global_batch = np.stack([state['global_in'][:-3] for state in obs])
    else:
        global_batch = np.stack([state['global_in'] for state in obs])
    return [global_batch]

def network_spec_irl(states, states_n, act, with_action, actions_size):

    global_state = states[0]
    global_state_n = states_n[0]
    action_state = tf.cast(act, tf.int32)

    # Jump
    agent_plane_x, agent_plane_z, agent_jump, is_grounded, can_double_jump, threedgrid = \
        tf.split(global_state, [1, 1, 1, 1, 1, 9261], axis=1)

    agent_plane_x = ((agent_plane_x + 1) / 2) * 500
    agent_plane_x = tf.cast(agent_plane_x, tf.int32)

    agent_plane_z = ((agent_plane_z + 1) / 2) * 500
    agent_plane_z = tf.cast(agent_plane_z, tf.int32)

    agent_jump = ((agent_jump + 1) / 2) * 40
    agent_jump = tf.cast(agent_jump, tf.int32)

    agent = tf.concat([agent_plane_x, agent_plane_z, agent_jump], axis=1)
    global_state = agent

    # agent_n_plane_x, agent_n_plane_z, agent_n_jump, _, _, _, _, _, _, _, _ = \
    #     tf.split(global_state_n, [1, 1, 1, 1, 1, 3, 2, 3375, 4, 12, 2], axis=1)
    #
    # agent_n_plane_x = ((agent_n_plane_x + 1) / 2) * 220
    # agent_n_plane_x = tf.cast(agent_n_plane_x, tf.int32)
    #
    # agent_n_plane_z = ((agent_n_plane_z + 1) / 2) * 280
    # agent_n_plane_z = tf.cast(agent_n_plane_z, tf.int32)
    #
    # agent_n_jump = ((agent_n_jump + 1) / 2) * 40
    # agent_n_jump = tf.cast(agent_n_jump, tf.int32)
    #
    # agent_n = tf.concat([agent_n_plane_x, agent_n_plane_z, agent_n_jump], axis=1)
    # global_state_n = agent_n

    # global_state = tf.compat.v1.Print(global_state, [global_state], 'Global state: ', summarize=1e5)
    # global_state = embedding(global_state, indices=280, size=32, name='embs')
    # global_state = tf.reshape(global_state, (-1, 3*32))
    # global_state = linear(global_state, 64, name='latent_1', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                       )
    threedgrid = tf.cast(tf.reshape(threedgrid, [-1, 21, 21, 21]), tf.int32)
    # threedgrid = tf.reshape(threedgrid, [-1, 15, 15, 15, 1])
    threedgrid_state = embedding(threedgrid, indices=4, size=32, name='global_embs')
    threedgrid = conv_layer_3d(threedgrid_state, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_01', activation=tf.nn.relu)
    #threedgrid = tf.nn.max_pool3d(threedgrid, [2, 2, 2], strides=(2, 2, 2), padding="VALID")
    threedgrid = conv_layer_3d(threedgrid, 32, [3, 3, 3], strides=(2, 2, 2), name='conv_02', activation=tf.nn.relu)
    #threedgrid = tf.nn.max_pool3d(threedgrid, [2, 2, 2], strides=(2, 2, 2), padding="VALID")
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_03', activation=tf.nn.relu)
    threedgrid = conv_layer_3d(threedgrid, 64, [3, 3, 3], strides=(2, 2, 2), name='conv_04', activation=tf.nn.relu)
    threedgrid = tf.reshape(threedgrid, [-1, 2 * 2 * 2 * 64])

    # global_state_n = embedding(global_state_n, indices=501, size=32, name='embs')
    # global_state_n = tf.reshape(global_state_n, (-1, 3 * 32))
    # global_state_n = linear(global_state_n, 64, name='latent_1_n', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                      )

    # action_state = tf.compat.v1.Print(action_state, [action_state], 'Action state: ', summarize=1e5)
    # action_state = tf.one_hot(action_state, 10)
    # action_state = tf.reshape(action_state, [-1, 10])
    # global_state = tf.one_hot(global_state, 280)
    # global_state = tf.reshape(global_state, [-1, 3*280])


    action_state = embedding(action_state, indices=10, size=512, name='action_embs')
    action_state = tf.reshape(action_state, [-1, 512])
    action = action_state
    # action_state = linear(action_state, 64, name='latent_action_n', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                      )


    # action_state = tf.compat.v1.layers.dropout(action_state, rate=0.2)

    # inventory = linear(inventory, 32, name='inventory_embs', activation=tf.nn.tanh)
    # inventory = linear(inventory, 64, name='latent_inventory_n', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                       )

    encoded = tf.concat([threedgrid, action], axis=1)

    global_state = linear(encoded, 1024, name='latent_1', activation=tf.nn.relu,

                         )

    global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.relu,

                          )

    global_state = linear(global_state, 128, name='latent_3', activation=tf.nn.relu,

                          )

    # global_state = linear(global_state, 512, name='latent_2', activation=tf.nn.relu,
    #                       init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
    #                                                                       dtype=tf.dtypes.float32)
    #                      )
    # global_state = tf.compat.v1.layers.dropout(global_state, rate=0.2)

    global_state = linear(global_state, 1, name='out',
                          init=tf.compat.v1.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=None,
                                                                          dtype=tf.dtypes.float32)
                          )
    # global_state = tf.compat.v1.layers.dropout(global_state, rate=0.2)



    return global_state, threedgrid_state, action_state
