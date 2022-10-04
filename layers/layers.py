import tensorflow as tf
from utils import *
from math import sqrt

## Layers
def linear(inp, inner_size, name='linear', bias=True, activation=None, init=None):
    with tf.compat.v1.variable_scope(name):
        lin = tf.compat.v1.layers.dense(inp, inner_size, name=name, activation=activation, use_bias=bias,
                                        kernel_initializer=init)
        return lin

def transformer(input, n_head, hidden_size, mask_value=None, mlp_layer=1, pooling='None',
                    residual=True, with_embeddings=False, with_ffn=False, post_norm=False, mask=None,
                    pre_norm=False, name='transformer', reuse=False):

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        # Utility functions
        # Compute q,k,v vectors
        def qkv_embed(input, heads, n_embd):
            if pre_norm:
                input = layer_norm(input, axis=3)

            qk = linear(input, hidden_size*2, name='qk')
            qk = tf.reshape(qk, (bs, T, NE, heads, n_embd // heads, 2))

            # (bs, T, NE, heads, features)
            query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]

            value = linear(input, hidden_size, name='v')
            value = tf.reshape(value, (bs, T, NE, heads, n_embd // heads))

            query = tf.transpose(query, (0, 1, 3, 2, 4),
                                 name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
            key = tf.transpose(key, (0, 1, 3, 4, 2),
                               name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
            value = tf.transpose(value, (0, 1, 3, 2, 4),
                                 name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)

            return query, key, value

        def self_attention(input, mask, heads, n_embd):
            query, key, value = qkv_embed(input, heads, n_embd)
            logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
            logits /= np.sqrt(n_embd / heads)
            softmax = stable_masked_softmax(logits, mask)

            att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)

            out = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
            n_output_entities = shape_list(out)[2]
            out = tf.reshape(out, (bs, T, n_output_entities, n_embd))  # (bs, T, n_output_entities, n_embd)

            return out, softmax

        def create_mask(input, value):
            '''
                Create mask from the input. If the first element is 99, then mask it.
                The mask must be 1 for the input and 0 for the
            '''

            # x = bs, NE, feature
            mask = 1 - tf.cast(tf.equal(input[:,:,:,0], value), tf.float32)
            return mask

        #Initialize
        input = input[:, tf.newaxis, :, :]

        bs, T, NE, features = shape_list(input)

        if mask != None or mask_value != None:
            if mask == None:
                mask = create_mask(input, mask_value)
            assert np.all(np.array(mask.get_shape().as_list()) == np.array(input.get_shape().as_list()[:3])), \
                f"Mask and input should have the same first 3 dimensions. {shape_list(mask)} -- {shape_list(input)}"
            mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)

        # Make a initial embeddings
        if with_embeddings:
            input = linear(input, hidden_size, activation=tf.nn.tanh, name='embs')

        # Get the output of the transformer (B, T, NE, F) and attention weights (B, T, NE, NE)
        a, att_weights = self_attention(input, mask, n_head, hidden_size)

        a = linear(a, hidden_size, name='mlp_0')

        if residual:
            a = input + a

        if with_ffn and mlp_layer > 3:
            for i in range(mlp_layer - 2):
                a = linear(a, hidden_size, name='mlp_{}'.format(i), activation=tf.nn.relu)
            a = linear(a, hidden_size, name='mlp_{}'.format(mlp_layer))

            if residual:
                a = a + input

        input = a

        if post_norm:
            input = layer_norm(input, axis=3)

        mask = tf.reshape(mask, (bs, T, NE))

        if pooling == 'avg':
            input = entity_avg_pooling_masked(input, mask)
            bs, T, features = shape_list(input)
            input = tf.reshape(input, (bs, features))
        elif pooling == 'max':
            input = entity_max_pooling_masked(input, mask)
            bs, T, features = shape_list(input)
            input = tf.reshape(input, (bs, features))

    return input, att_weights

def layer_norm(input_tensor, axis):
  """Run layer normalization on the axis dimension of the tensor."""
  layer_norma = tf.keras.layers.LayerNormalization(axis = axis)
  return layer_norma(input_tensor)

# Circular 1D convolution
def circ_conv1d(inp, **conv_kwargs):
    valid_activations = {'relu': tf.nn.relu, 'tanh': tf.tanh, '': None}
    assert 'kernel_size' in conv_kwargs, f"Kernel size needs to be specified for circular convolution layer."
    conv_kwargs['activation'] = valid_activations[conv_kwargs['activation']]

    # Add T to input
    inp = tf.expand_dims(inp, axis = 1)
    # Concatenate input for circular convolution
    kernel_size = conv_kwargs['kernel_size']
    num_pad = kernel_size // 2
    inp_shape = shape_list(inp)
    inp_rs = tf.reshape(inp, shape=[inp_shape[0] * inp_shape[1]] + inp_shape[2:]) #  (BS * T, NE, feats)
    inp_padded = tf.concat([inp_rs[..., -num_pad:, :], inp_rs, inp_rs[..., :num_pad, :]], -2)
    out = tf.compat.v1.layers.conv1d(inp_padded,
                           kernel_initializer=tf.initializers.GlorotUniform(),
                           padding='valid',
                           **conv_kwargs)

    out = tf.reshape(out, shape=inp_shape[:3] + [conv_kwargs['filters']])
    return out

def conv_layer_2d(input, filters, kernel_size, strides=(1, 1), padding="SAME", name='conv',
                  activation=None, bias=True):

    with tf.compat.v1.variable_scope(name):
        conv = tf.compat.v1.layers.conv2d(input, filters, kernel_size, strides, padding=padding, name=name,
                                          activation=activation, use_bias=bias)
        return conv

def conv_layer_3d(input, filters, kernel_size, strides=(1, 1, 1), padding="SAME", name='conv',
                  activation=None, bias=True):

    with tf.compat.v1.variable_scope(name):
        conv = tf.compat.v1.layers.conv3d(input, filters, kernel_size, strides, padding=padding, name=name,
                                          activation=activation, use_bias=bias)
        return conv

def embedding(input, indices, size, name='embs'):
    with tf.compat.v1.variable_scope(name):
        shape = (indices, size)
        stddev = min(0.1, sqrt(2.0 / (product(xs=shape[:-1]) + shape[-1])))
        initializer = tf.random.normal(shape=shape, stddev=stddev, dtype=tf.float32)
        W = tf.Variable(
            initial_value=initializer, trainable=True, validate_shape=True, name='W',
            dtype=tf.float32, shape=shape
        )
        return tf.nn.tanh(tf.compat.v1.nn.embedding_lookup(params=W, ids=input, max_norm=None))

def create_mask(input, value):
    '''
        Create mask from the input. If the first element is 99, then mask it.
        The mask must be 1 for the input and 0 for the
    '''
    # x = bs, NE, feature
    input = input[:, tf.newaxis, :, :]
    mask = 1 - tf.cast(tf.equal(input[:,:,:,0], value), tf.float32)
    return mask

def positional_encoding(x, dimension):
    d = tf.range(dimension)[tf.newaxis, ...]
    def get_angles(pos, i, d_model):
        pos = tf.cast(pos, tf.float32)
        angle_rates = 1 / tf.pow(tf.cast(10000, tf.float32), tf.cast((2 * (i // 2)) / d_model, tf.float32))
        return pos * angle_rates

    angles = get_angles(x, d, dimension)
    sins = tf.math.sin(angles[:,0::2])
    coss = tf.math.cos(angles[:,1::2])

    sins = tf.expand_dims(sins, 2)
    coss = tf.expand_dims(coss, 2)

    embs = tf.concat([sins, coss], 2)

    embs = tf.reshape(embs, [-1, dimension])
    return embs