# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import max_pool_2d
from tflearn.layers.estimator import regression

import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sisws import spatial_weight_sharing

import os
import numpy as np

import scipy.misc
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.ERROR)

COLOR_CODING = True
if COLOR_CODING:
    try:
        import colorlover as cl
    except ImportError:
        print("WARNING: Unable to import colorlover, you can install it through 'pip install colorlover --user'\n"
              "For now, this layer does not use color coding")
        distance_summ = False

project_folder = os.path.dirname(os.path.realpath(__file__))

centroid_grid = [2, 2]
n_centroids = np.prod(centroid_grid)

X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
#network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = spatial_weight_sharing(network, centroid_grid, 24, 7, 1, tf.nn.relu, centroids_trainable=True, scaling=1.,
                                 distance_fn='EXP', local_normalization=True, sigma_trainable=True, per_feature=False,
                                 color_coding=True)
visiual_summary = network.visual_summary
bias = network.b
W_list = network.W_list
network = max_pool_2d(network, 2)
#network = local_response_normalization(network)
#network = conv_2d(network, 64, 3, activation='linear', regularizer="L2")
#network = spatial_weight_sharing(network, [3, 3], 48, 3, 1, tf.nn.relu, centroids_trainable=True, scaling=.001,
#                                 distance_fn='EXP', local_normalization=True, sigma_trainable=True)
#network = max_pool_2d(network, 2)
#network = local_response_normalization(network)
#network = spatialsoftmax(network)

#network = spatial_weight_sharing(network, [2, 2], 48, 3, 1, tf.nn.relu, centroids_trainable=True, scaling=1,
#                                 distance_fn='EXP', local_normalization=True, sigma_trainable=False, per_feature=True)
network = fully_connected(network, 128, activation='relu')
#network = fully_connected_weight_sharing(network, 11, 3, dimensionality='square')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='relu')
#network = fully_connected_weight_sharing(network, 15, 3, dimensionality='square')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x )

# Training

kernel_summ = tf.summary.image("Locally weighted filters", visiual_summary, max_outputs=24)

model = tflearn.DNN(network, tensorboard_verbose=0,
                    tensorboard_dir=os.path.join(project_folder, 'tensorboard'))

model.fit({'input': X}, {'target': Y}, n_epoch=1,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')



#image_sum = tf.summary.image("SpatialWeightSharingFilters", visiual_summary)
weighted_filters, summ = model.session.run([visiual_summary, kernel_summ])

#print(summ)
if COLOR_CODING:
    colors = [(c[0] / 255., c[1] / 255., c[2] / 255., 1.)
              for c in cl.to_numeric(cl.scales[str(max(3, n_centroids))]['div']['Spectral'])[:n_centroids]]

    patches = [mpatches.Patch(color=c, label='Centroid {}'.format(i)) for i, c in enumerate(colors)]
    fig = plt.figure(figsize=(2, 2))
    fig.legend(handles=patches, labels=['Centroid {}'.format(i) for i in range(n_centroids)])
    plt.savefig('tmp.png')
    plot_image = scipy.misc.imread('tmp.png', mode='RGB')
    plt.imshow(plot_image)

    first_filter = tf.expand_dims(tf.constant(plot_image.astype('float') / 255., dtype=tf.float32), 0)
    im = tf.image.resize_nearest_neighbor(tf.constant(weighted_filters[1:, :, :, :3], dtype=tf.float32), plot_image.shape[:2])
    kernel_summ = tf.summary.image("Locally weighted with colors", tf.concat(0, [first_filter, im]), max_outputs=24)
    kernel_grayscale_summ = tf.summary.image("Locally weighted grayscale", tf.constant(weighted_filters[:, :, :, -1:]),
                                             max_outputs=24)
    distance_summ = tf.summary.image("Locally weighted distance", tf.constant(weighted_filters), max_outputs=24)
    summaries = model.session.run([kernel_summ, kernel_grayscale_summ, distance_summ, im])
else:
    summaries = [summ]

model.trainer.summ_writer.reopen()
for s in summaries[:-1]:
    model.trainer.summ_writer.add_summary(s)
#if COLOR_CODING:
#    model.trainer.summ_writer.add_summary(summ, step)
model.trainer.summ_writer.close()

Ws = model.session.run(W_list)
print("Biases: \n{}\n".format(model.session.run(bias)[-9*2:-9]))
print("Sigmas: \n{}\n".format(model.session.run(bias)[-9:]))
for i in range(weighted_filters.shape[0]):
    #image = Image.fromarray(weighted_filters[i, :, :, 0]).convert('RGB')
    #image.save('mnist_logs/weighted_filters{0}.png'.format(i), format='png')
    scipy.misc.imsave(os.path.join(project_folder, 'kernel_images', 'weighted_filters{0}.png'.format(i)),
                      weighted_filters[i, :, :, :])

    for j, W in enumerate(Ws):
        scipy.misc.imsave(os.path.join(project_folder, 'kernel_images',
                                       'plain_filter{0}_{1}.png'.format(i, j)), W[:, :, 0, i])
