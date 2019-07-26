# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

"""Small repro case for keras batchnorm slowdowns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import keras
from keras.layers import BatchNormalization, Conv2D
import tensorflow as tf


BATCH_SIZE = 16
INPUT_H = 512
INPUT_W = 512
OUTPUT_H = 32
OUTPUT_W = 32
NUM_OUTPUT_CHANNELS = 16

PADDING = 'same'
DATA_FORMAT = 'channels_first'


# We'll be patching Keras's tensorflow backend with this version to avoid the issue mentioned here:
# https://github.com/keras-team/keras/issues/10382
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / sqrt(var + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        axis: Integer, the axis that should be normalized.
            (typically the features axis).
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


def _patch_backend_function(f):
    name = f.__name__
    keras.backend.__setattr__(name, f)
    keras.backend.tensorflow_backend.__setattr__(name, f)


def get_model(input_layer, use_batch_norm):
    """Get a reasonably beefy model to better demonstrate the slowdown."""
    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=PADDING,
        data_format=DATA_FORMAT,
        )(input_layer)
    if use_batch_norm:
        x = BatchNormalization(axis=1)(x)

    x = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=PADDING,
        data_format=DATA_FORMAT,
        )(x)
    if use_batch_norm:
        x = BatchNormalization(axis=1)(x)

    x = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=PADDING,
        data_format=DATA_FORMAT,
        )(x)
    if use_batch_norm:
        x = BatchNormalization(axis=1)(x)

    x = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=PADDING,
        data_format=DATA_FORMAT,
        )(x)
    if use_batch_norm:
        x = BatchNormalization(axis=1)(x)

    # Output.
    x = Conv2D(
        filters=NUM_OUTPUT_CHANNELS,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=PADDING,
        data_format=DATA_FORMAT,
        )(x)

    model = keras.models.Model(
        inputs=input_layer,
        outputs=x)

    return model


class UpdateHook(tf.train.SessionRunHook):
    """This hook is a lightweight version of one used in practice.

    This hook is typically needed for various things: making sure the keras backend session is
    the same as that being used by TensorFlow, initializing a keras model's weights to be the same
    as those from the TF graph, and running various model updates (e.g. batch norm moving average
    and variance).
    """

    def __init__(self, keras_model):
        # This would be where batch norm updates come from.
        self._updates = keras_model.updates

    def after_create_session(self, session, coord):
        """Called when a new TF session is created."""
        keras.backend.set_session(session)

    def before_run(self, run_context):
        """Called before each call to session.run."""
        return tf.train.SessionRunArgs(self._updates)


def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


def parse_command_line_arguments(args=None):
    """Parse command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--infer',
        action='store_true',
        help='If true, the keras model\'s __call__ method will be called on the inputs that '
             'were used to define its input layer.',
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_command_line_arguments()

    # Patch keras backend function.
    _patch_backend_function(batch_normalization)

    # Set keras backend format.
    keras.backend.set_image_data_format(DATA_FORMAT)

    # Get random input.
    inputs = tf.random_normal(shape=[BATCH_SIZE, 3, INPUT_H, INPUT_W])

    model_input = keras.layers.Input(tensor=inputs)
    model = get_model(input_layer=model_input, use_batch_norm=True)

    y_true = tf.random_normal(shape=[BATCH_SIZE, NUM_OUTPUT_CHANNELS, OUTPUT_H, OUTPUT_W])

    if args.infer:
        # This leads to the slowdown.
        y_pred = model(inputs)
    else:
        y_pred = model.outputs[0]

    loss = loss_fn(y_pred, y_true)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)

    minimize_op = optimizer.minimize(loss)

    init_op = tf.group(tf.local_variables_initializer())
    scaffold = tf.train.Scaffold(local_init_op=init_op)

    hooks = [UpdateHook(keras_model=model)]

    with tf.train.SingularMonitoredSession(scaffold=scaffold, hooks=hooks) as session:
        start = time.time()
        for i in range(1000):
            loss_value, _ = session.run([loss, minimize_op])
            if i % 100 == 0:
                end = time.time()
                print("Step %d: %.6f" % (i, loss_value))
                print("Took %.4f" % (end - start))
                start = time.time()
