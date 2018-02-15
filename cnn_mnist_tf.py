"""TensorFlow MNIST Classifier: https://www.tensorflow.org/tutorials/layers"""

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
BATCH_SIZE = 100
NUM_EPOCHS = 5
NUM_CLASSES = 10


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    
    train = mode == tf.estimator.ModeKeys.TRAIN
    predict = mode == tf.estimator.ModeKeys.PREDICT

    input_layer = tf.reshape(features['x'], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS])
    conv_layer_1 = tf.layers.conv2d(inputs=input_layer,
                                    filters=32,
                                    kernel_size=[5, 5],
                                    padding='same',
                                    activation=tf.nn.relu)
    pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 2], strides=2)
    conv_layer_2 = tf.layers.conv2d(inputs=pool_layer_1,
                                    filters=64,
                                    kernel_size=[5, 5],
                                    padding='same',
                                    activation=tf.nn.relu)
    pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 2], strides=2)
    # pool 1 -> (14, 14, 32), pool 2 -> (7, 7, 64)
    flat_layer = tf.reshape(pool_layer_2, [BATCH_SIZE, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=flat_layer, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=train)
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    predictions = dict(classes=tf.argmax(input=logits, axis=1),
                       probabilities=tf.nn.softmax(logits, name='softmax_tensor'))

    if predict:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if train:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = dict(accuracy=tf.metrics.accuracy(labels=labels, predictions=predictions['classes']))
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = mnist.train.labels.astype(np.int32)
    test_data = mnist.test.images
    test_labels = mnist.test.labels.astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='./models/mnist_convnet_model')
    tensors_to_log = dict(probabilities='softmax_tensor')
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=dict(x=train_data), 
                                                        y=train_labels, 
                                                        batch_size=BATCH_SIZE, 
                                                        num_epochs=None, 
                                                        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=BATCH_SIZE * NUM_EPOCHS, hooks=[logging_hook])

    # eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=dict(x=test_data),
                                                       y=test_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    test_labels = mnist_classifier.evaluate(input_fn=eval_input_fn)


if __name__ == '__main__':
    tf.app.run()
