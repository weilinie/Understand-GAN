import os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def load_dataset(data_path, batch_size, scale_size, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(data_path)
    if dataset_name in ['CelebA'] and split:
        data_path = os.path.join(data_path, 'splits', split)
    elif dataset_name in ['RenderBall', 'RenderBallTri']:
        data_path = data_path
    else:
        is_grayscale = True
        raise Exception('[!] Caution! Unknown dataset name.')

    paths = []
    tf_decode = tf.image.decode_jpeg
    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(data_path, ext))

        if ext == 'png':
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
        shape = [h, w, 1]
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    return tf.to_float(queue)


def load_mnist(data_path):
    mnist_data = read_data_sets(data_path, one_hot=True)

    return mnist_data


def generate_gmm_circle_data(num_data=50000, dim=2, num_cluster=8, scale=4, var=0.02):
    means_x = np.array([scale * np.cos(i * 2 * np.pi / num_cluster) for i in range(num_cluster)])
    means_y = np.array([scale * np.sin(i * 2 * np.pi / num_cluster) for i in range(num_cluster)])
    means = np.vstack((means_x, means_y)).transpose()
    # print means
    std = np.array([var] * num_cluster).transpose()
    weights = np.array([1. / num_cluster] * num_cluster).transpose()
    if num_cluster == 2:
        weights = np.array([2./3, 1./3])

    data = np.zeros([num_data, 2], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(2) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster

    data = np.clip(data, -3 * scale, 3 * scale)

    return data, means


def batch_gmm_gen(data, batch_size):
    while True:
        np.random.shuffle(data)
        for i in range(0, len(data) - batch_size + 1, batch_size):
            yield data[i:i + batch_size]

