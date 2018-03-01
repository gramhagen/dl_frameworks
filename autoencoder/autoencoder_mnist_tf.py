"""TensorFlow MNIST Classifier: https://www.tensorflow.org/tutorials/layers"""

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


tf.logging.set_verbosity(tf.logging.INFO)
mnist = tf.contrib.learn.datasets.load_dataset('mnist')

BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10
MODEL_DIR = 'models'


def ae_model_fn(features, labels, mode):
    """Model function for Autoencoder"""

    input_layer = tf.reshape(features['x'], [-1, 784])
    layer_1 = tf.layers.dense(inputs=input_layer, units=128, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu)
    layer_3 = tf.layers.dense(inputs=layer_2, units=32, activation=tf.nn.relu)
    embeddings = tf.layers.dense(inputs=layer_3, units=16, activation=tf.nn.relu)
    layer_5 = tf.layers.dense(inputs=embeddings, units=32, activation=tf.nn.relu)
    layer_6 = tf.layers.dense(inputs=layer_5, units=64, activation=tf.nn.relu)
    layer_7 = tf.layers.dense(inputs=layer_6, units=128, activation=tf.nn.relu)
    predictions = tf.layers.dense(inputs=layer_7, units=784, activation=tf.nn.sigmoid)


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=embeddings)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = dict(accuracy=tf.metrics.accuracy(labels=labels, predictions=predictions))
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_data = mnist.train.images
    test_data = mnist.test.images

    autoencoder = tf.estimator.Estimator(model_fn=ae_model_fn, model_dir=MODEL_DIR)
    logging_hook = tf.train.LoggingTensorHook(tensors=dict(), every_n_iter=50)

    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=dict(x=train_data), 
                                                        y=train_data,
                                                        batch_size=BATCH_SIZE, 
                                                        num_epochs=None, 
                                                        shuffle=True)
    autoencoder.train(input_fn=train_input_fn, steps=BATCH_SIZE * NUM_EPOCHS, hooks=[logging_hook])

    # eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=dict(x=test_data),
                                                       y=test_data,
                                                       shuffle=False)
    score = autoencoder.evaluate(input_fn=eval_input_fn)
    print(score)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=dict(x=test_data),
                                                          batch_size=len(test_data),
                                                          shuffle=False)
    embeddings = autoencoder.predict(input_fn=predict_input_fn)

    with tf.Session() as sess:
        # pull the embeddings into a tf tensor and initialize variables
        embeddings_tf = tf.Variable(tf.stack([e for e in embeddings], axis=0), name="embeddings")
        tf.global_variables_initializer().run()
        # save the tensor down
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, MODEL_DIR + '/embeddings', 0)
        # associate metadata with the embedding
        summary_writer = tf.summary.FileWriter(MODEL_DIR + '/embeddings')
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings_tf.name
        embedding.metadata_path = 'metadata.tsv'
        # save a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    tf.app.run()
