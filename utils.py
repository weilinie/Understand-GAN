from __future__ import print_function

import functools
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from scipy.misc import imsave
import sklearn.mixture
import collections

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_since_beginning = collections.defaultdict(lambda: {})


def prepare_dirs(config, dataset):
    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(dataset, datetime.now().strftime("%m%d_%H%M%S"))

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if not hasattr(config, 'data_path'):
        config.data_path = os.path.join(config.data_dir, dataset)

    for dir in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if not config.is_train:
        if not hasattr(config, 'test_model_dir'):
            config.test_model_dir = os.path.join(config.test_dir, config.model_name)
        if not os.path.exists(config.test_model_dir):
            os.makedirs(config.test_model_dir)


# language dataset iterator
def inf_train_gen(lines, batch_size, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - batch_size + 1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + batch_size]],
                dtype=np.int32
            )


# image residual block
def resBlock(inputs, input_num, output_num, kernel_size, resample=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(tcl.conv2d, stride=2)
        conv_1 = functools.partial(tcl.conv2d, num_outputs=input_num / 2)
        conv_1b = functools.partial(tcl.conv2d, num_outputs=output_num / 2, stride=2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)
    elif resample == 'up':
        conv_shortcut = subpixelConv2D
        conv_1 = functools.partial(tcl.conv2d, num_outputs=input_num / 2)
        conv_1b = functools.partial(tcl.conv2d_transpose, num_outputs=output_num / 2, stride=2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)
    elif resample == None:
        conv_shortcut = tcl.conv2d
        conv_1 = functools.partial(tcl.conv2d, num_outputs=input_num / 2)
        conv_1b = functools.partial(tcl.conv2d, num_outputs=output_num / 2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)

    else:
        raise Exception('invalid resample value')

    if output_num == input_num and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs=inputs, num_outputs=output_num, kernel_size=1)  # Should kernel_size be larger?

    output = inputs
    output = conv_1(inputs=output, kernel_size=1)
    output = conv_1b(inputs=output, kernel_size=kernel_size)
    output = conv_2(inputs=output, kernel_size=1, biases_initializer=None)  # Should skip bias here?
    # output = Batchnorm(name+'.BN', [0,2,3], output) # Should skip BN op here?

    return shortcut + (0.3 * output)


# use depth-to-space for upsampling image
def subpixelConv2D(*args, **kwargs):
    kwargs['num_outputs'] = 4 * kwargs['num_outputs']
    output = tcl.conv2d(*args, **kwargs)
    output = tf.depth_to_space(output, 2)
    return output


def f_congugate(t, option="KL", alpha=0):
    if option == "KL":
        return tf.exp(t - 1)
    elif option == "RKL":
        return -1 - tf.log(-t)
    elif option == "JS":
        return -tf.log(2 - tf.exp(t))
    elif option == "Hellinger":
        return t / (1 - t)
    elif option == "TV":
        return t
    elif option == "Pearson":
        return t ** 2 / 4 + t
    elif option == "alpha" and alpha != 0:
        return 1 / alpha * ((1 - alpha) * t + 1) ** (alpha / (alpha - 1)) - 1 / alpha
    else:
        raise Exception("Not implemented divergence option")


def g_f(v, option="KL", alpha=0.5):
    if option == "KL":
        return v
    elif option == "RKL":
        return -tf.exp(v)
    elif option == "JS":
        return tf.log(2) - tf.log(1 + tf.exp(-v))
    elif option == "Hellinger":
        return 1 - tf.exp(-v)
    elif option == "TV":
        return tf.tanh(v) / 2
    elif option == "Pearson":
        return v
    elif option == "alpha" and alpha != 0:
        if alpha < 1:
            return 1 / (1 - alpha) - tf.log(1 + tf.exp(-v))
        else:
            return v
    else:
        raise Exception("Not implemented divergence option")


def make_grid(tensor, nrow=8, padding=2):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    batch_size = tensor.shape[0]
    xmaps = min(nrow, batch_size)
    ymaps = batch_size // xmaps
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + padding, width * xmaps + padding, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= batch_size:
                break
            h, h_width = y * height + padding, height - padding
            w, w_width = x * width + padding, width - padding

            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k += 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding)
    # im = Image.fromarray(ndarr)
    imsave(filename, ndarr)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def layer_norm(inputs):
    ndims_inputs = inputs.get_shape().ndims

    mean, var = tf.nn.moments(inputs, range(1, ndims_inputs), keep_dims=True)

    # Assume the 'neurons' axis is the last of norm_axes. This is the case for fully-connected and NHWC conv layers.
    n_neurons = inputs.get_shape().as_list()[ndims_inputs - 1]

    offset = tf.Variable(np.zeros(n_neurons, dtype='float32'), name='offset')
    scale = tf.Variable(np.ones(n_neurons, dtype='float32'), name='scale')

    # Add broadcasting dims to offset and scale (e.g. NHWC conv data)
    offset = tf.reshape(offset, [1 for _ in range(ndims_inputs - 1)] + [-1])
    scale = tf.reshape(scale, [1 for _ in range(ndims_inputs - 1)] + [-1])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result


# Use for computing real_nearby_gradient_penalty
def get_perturbed_batch(minibatch):
    _, var = tf.nn.moments(minibatch, axes=[i for i in range(0, minibatch.shape.ndims)])
    return minibatch + 0.5 * tf.sqrt(var) * np.random.random(minibatch.shape)


# Plot inception score
def plot_incept_score(idx, incept_score, save_step, model_dir):
    _since_beginning[idx] = incept_score
    if idx % save_step == 0:
        x_vals = np.sort(_since_beginning.keys())
        y_vals = [_since_beginning[x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel('inception score')
        plt.savefig(os.path.join(model_dir, 'inception_score.jpg'))


# spherical linear interplotion
def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


# save GMM results
def save_gmm(data, save_path, center=None, title=None, is_color=True):
    """
        data is a N * M matrix, where N is number of data, M is number of features (2)
        """
    data = np.reshape(data, (data.shape[0], -1))
    s_axis = np.linalg.norm(center[0]) * 2
    axis = [-s_axis, s_axis, -s_axis, s_axis]

    plt.close('all')
    plt.figure()

    if is_color:
        _, label, _ = estimate_optimal_cluster_size_gmm(
            data, clusters=range(1, center.shape[0] + 1)
        )
        if not np.max(label) == 0:
            colors = label / (np.max(label) + 0.0)
        else:
            colors = label
        plt.scatter(data[:, 0], data[:, 1], c=colors, marker="+")
    else:
        plt.scatter(data[:, 0], data[:, 1], marker="+")
    if not (center == None):
        plt.scatter(center[:, 0], center[:, 1], marker=(5, 0), c="k")
    if not (axis == None):
        plt.axis(axis)
    plt.grid(True)
    if not (title == None):
        plt.title(title)
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)


def estimate_optimal_cluster_size_gmm(data, clusters=range(1, 10), run=10):
    out = []
    predict = []
    gmms = []
    # calculate transformed distortion
    for c in clusters:
        gmm = sklearn.mixture.GaussianMixture(n_components=c, init_params='kmeans', n_init=run, covariance_type='full')
        # gmm = sklearn.mixture.VBGMM(n_components=c)
        gmm.fit(data)
        gmms.append(gmm)
        out.append(-gmm.bic(data))
        predict.append(gmm.predict(data))
        # find optimal cluster count
    # print out
    i = np.argmax(out)
    # print(out)
    return clusters[i], predict[i], gmms[i]
