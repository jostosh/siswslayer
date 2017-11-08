# -*- coding: utf-8 -*-

""" Convolutional Neural Network using spatial interpolation soft weight sharing for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""
import tflearn
from tflearn.layers.normalization import local_response_normalization
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
import tflearn.datasets.mnist as mnist
from tensorflow.python.platform import tf_logging as logging
import argparse


logging.set_verbosity(logging.ERROR)                            # Here we suppress tflearn warnings
project_folder = os.path.dirname(os.path.realpath(__file__))


def prepare_data():
    """
    Prepares MNIST data
    """
    X_train, Y_train, X_test, Y_test = mnist.load_data(one_hot=True)
    X_train = X_train.reshape([-1, 28, 28, 1])
    X_test = X_test.reshape([-1, 28, 28, 1])
    return X_train, Y_train, X_test, Y_test


def save_kernel_figs(session, sws_layer, weighted_filters):
    """
    Saves figures of the separate KCP kernels and the spatially weighted kernels
        :param session:             The TensorFlow session
        :param sws_layer:           The sisws layer op
        :param weighted_filters:    The spatially weighted kernels
    """
    os.makedirs(os.path.join(project_folder, 'kernel_images'), exist_ok=True)
    Ws = session.run(sws_layer.W_list)
    for i in range(weighted_filters.shape[0]):
        scipy.misc.imsave(os.path.join(project_folder, 'kernel_images', 'weighted_filters{0}.png'.format(i)),
                          weighted_filters[i, :, :, -1])

        for j, W in enumerate(Ws):
            scipy.misc.imsave(os.path.join(project_folder, 'kernel_images',
                                           'plain_filter{0}_{1}.png'.format(i, j)), W[:, :, 0, i])


def create_legend(colors, n_centroids):
    """
    Creates a simple legend to clarify the color coding in TensorBoard
        :param colors:          The colors to visualize
        :param n_centroids:     The number of centroids

    Returns:
        :return: An image containing a legend (generated with matplotlib)
    """
    patches = [mpatches.Patch(color=c, label='Centroid {}'.format(i)) for i, c in enumerate(colors)]
    fig = plt.figure(figsize=(3, 3))
    fig.legend(handles=patches, labels=['Centroid {}'.format(i) for i in range(n_centroids)])
    plt.savefig('tmp.png')
    plot_image = scipy.misc.imread('tmp.png', mode='RGB')
    os.remove('tmp.png')
    return plot_image


def sws_visualization(args, kernel_summ, model, n_centroids, visual_summary):
    """
    Here we create a visualization of the spatially weighted kernels
        :param args: The arguments passed at initiating the script
        :param kernel_summ: The kernel summary
        :param model: The tflearn model
        :param n_centroids: The number of centroids
        :param visual_summary: The visual summary op
    Returns:
        :return: weighted_filters: The spatially weighted filters
    """
    weighted_filters, summ = model.session.run([visual_summary, kernel_summ])
    if args.color_coding:
        # In this case we export color summaries to TensorBoard
        colors = [(c[0] / 255., c[1] / 255., c[2] / 255., 1.)
                  for c in cl.to_numeric(cl.scales['9']['qual']['Set1'])[:n_centroids]]

        legend_image = create_legend(colors, n_centroids)
        first_filter = tf.expand_dims(tf.constant(legend_image.astype('float') / 255., dtype=tf.float32), 0)
        im = tf.image.resize_nearest_neighbor(tf.constant(weighted_filters[1:, :, :, :3], dtype=tf.float32),
                                              legend_image.shape[:2])
        kernel_summ = tf.summary.image("Locally weighted with colors", tf.concat(0, [first_filter, im]), max_outputs=24)
        kernel_grayscale_summ = tf.summary.image("Locally weighted grayscale",
                                                 tf.constant(weighted_filters[:, :, :, -1:]),
                                                 max_outputs=24)
        distance_summ = tf.summary.image("Locally weighted distance", tf.constant(weighted_filters), max_outputs=24)
        summaries = model.session.run([kernel_summ, kernel_grayscale_summ, distance_summ])
    else:
        # Otherwise, our summaries are given by just the kernels themselves
        summaries = [summ]
    model.trainer.summ_writer.reopen()
    for s in summaries:
        model.trainer.summ_writer.add_summary(s)
    model.trainer.summ_writer.close()
    return weighted_filters


def build_cnn(args, n_centroids):
    """
    Builds CNN with 2 spatial interpolation soft weight sharing layers
        :param args: Command line arguments
        :param n_centroids: The number of centroids
    Returns
        :return: network: An Op that defines the head of the network
                 sws_layer: The first sisws layer
    """
    n_filters = args.n_filters
    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = spatial_weight_sharing(incoming=network, n_centroids=n_centroids, n_filters=n_filters[0], filter_size=7,
                                     strides=1, activation=tf.nn.relu, centroids_trainable=args.centroids_trainable,
                                     per_feature=True, color_coding=args.color_coding, similarity_fn='InvEuclidean')
    sws_layer = network
    network = local_response_normalization(network)
    network = max_pool_2d(network, 2)
    network = spatial_weight_sharing(incoming=network, n_centroids=n_centroids, n_filters=n_filters[1], filter_size=3,
                                     strides=1, activation=tf.nn.relu, centroids_trainable=args.centroids_trainable,
                                     per_feature=True, similarity_fn='InvEuclidean')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    return network, sws_layer


def main(args):
    """
    Performs training on the MNIST dataset with spatial interpolation soft weight sharning layers
    :param args: Command line argumentss
    """
    X_train, Y_train, X_test, Y_test = prepare_data()
    n_centroids = args.centroid_grid if not args.n_centroids else args.n_centroids
    # Building convolutional network
    network, sws_layer = build_cnn(args, n_centroids)
    # Create a kernel summary that will be the default visualization of the locally weighted kernels of the spatial
    # weight sharing layer
    visual_summary = sws_layer.visual_summary
    kernel_summ = tf.summary.image("Locally weighted filters", visual_summary, max_outputs=args.n_filters[0])

    # Use tflearns DNN to create a model
    model = tflearn.DNN(network, tensorboard_verbose=args.log_verbosity, tensorboard_dir=args.logdir)
    n_centroids = np.prod(n_centroids)
    model.fit({'input': X_train}, {'target': Y_train}, n_epoch=args.n_epochs,
              validation_set=({'input': X_test}, {'target': Y_test}),
              snapshot_step=100, show_metric=True, run_id='convnet_mnist')

    # Store nice visualizations for TensorBoard
    weighted_filters = sws_visualization(args, kernel_summ, model, n_centroids, visual_summary)
    # Also store images of the kernels themselves
    save_kernel_figs(model.session, sws_layer, weighted_filters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration of the soft spatial weight sharing layer")
    parser.add_argument("--color_coding", dest='color_coding', action='store_true', default=True,
                        help='Whether to use color coding in TensorBoard visualizations')
    parser.add_argument("--centroid_grid", nargs='+', type=int, default=[2, 2], help='Grid in which the centroids are '
                                                                                     'arranged at initialization')
    parser.add_argument("--n_centroids", type=int, default=None, help='If n_centroids is given, the centroids are '
                                                                      'initialized randomly')
    parser.add_argument("--logdir", default=os.path.join(project_folder, 'tensorboard'),
                        help='Specify dir for TensorFlow logs')
    parser.add_argument("--centroids_trainable", dest='centroids_trainable', default=False, action='store_true',
                        help='If given, the centroid positions will be trainable parameters')
    parser.add_argument("--log_verbosity", type=int, default=0, help="TensorBoard log verbosity")
    parser.add_argument("--n_filters", nargs='+', type=int, default=[24, 48],
                        help="Number of filters in the conv layers.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    if args.color_coding:
        try:
            import colorlover as cl
        except ImportError:
            print("WARNING: Unable to import colorlover, you can install it through 'pip install colorlover --user'\n"
                  "For now, this layer does not use color coding")
            args.color_coding = False

    main(args)