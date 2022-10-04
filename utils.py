import numpy as np
import json
import tensorflow as tf

def product(xs, empty=1):
    result = None
    for x in xs:
        if result is None:
            result = x
        else:
            result *= x

    if result is None:
        result = empty

    return result

class RunningStat(object):
        def __init__(self, shape=()):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM)/self._n
                self._S[...] = self._S + (x - oldM)*(x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            if self._n >= 2:
                return self._S/(self._n - 1)
            else:
                return np.square(self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):

            return self._M.shape

class LimitedRunningStat(object):
    def __init__(self, len=1000):
        self.values = np.array(np.zeros(len))
        self.n_values = 0
        self.i = 0
        self.len = len

    def push(self, x):
        self.values[self.i] = x
        self.i = (self.i + 1) % len(self.values)
        if self.n_values < len(self.values):
            self.n_values += 1

    @property
    def n(self):
        return self.n_values

    @property
    def mean(self):
        return np.mean(self.values[:self.n_values])

    @property
    def var(self):
        return np.var(self.values[:self.n_values])

    @property
    def std(self):
        return np.std(self.values[:self.n_values])

class DynamicRunningStat(object):

    def __init__(self):
        self.current_rewards = list()
        self.next_rewards = list()

    def push(self, x):
        self.next_rewards.append(x)

    def reset(self):
        self.current_rewards = self.next_rewards
        self.next_rewards = list()

    @property
    def n(self):
        return len(self.current_rewards)

    @property
    def mean(self):
        return np.mean(np.asarray(self.current_rewards))

    @property
    def std(self):
        return np.std(np.asarray(self.current_rewards))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def shape_list(x):
    '''
        deal with dynamic shape in tensorflow cleanly
    '''
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def stable_masked_softmax(logits, mask):

    #  Subtract a big number from the masked logits so they don't interfere with computing the max value
    if mask is not None:
        mask = tf.expand_dims(mask, 2)
        logits -= (1.0 - mask) * 1e10

    #  Subtract the max logit from everything so we don't overflow
    logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
    unnormalized_p = tf.exp(logits)

    #  Mask the unnormalized probibilities and then normalize and remask
    if mask is not None:
        unnormalized_p *= mask
    normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
    if mask is not None:
        normalized_p *= mask
    return normalized_p

def entity_avg_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    masked = x * mask
    summed = tf.reduce_sum(masked, -2)
    denom = tf.reduce_sum(mask, -2) + 1e-5
    return summed / denom

def entity_max_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    has_unmasked_entities = tf.sign(tf.reduce_sum(mask, axis=-2, keepdims=True))
    offset = (mask - 1) * 1e9
    masked = (x + offset) * has_unmasked_entities
    return tf.reduce_max(masked, -2)

# Boltzmann transformation to probability distribution
def boltzmann(probs, temperature = 1.):
    sum = np.sum(np.power(probs, 1/temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.power(p, 1/temperature) / sum)

    return np.asarray(new_probs)

# Very fast np.random.choice
def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]