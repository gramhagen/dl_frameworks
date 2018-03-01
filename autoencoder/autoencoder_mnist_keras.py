"""Keras MNIST Autoencoder: https://blog.keras.io/building-autoencoders-in-keras.html"""

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.callbacks import TensorBoard
from time import time
import tensorflow as tf


BATCH_SIZE = 256
NUM_EPOCHS = 100
NUM_CLASSES = 10


if __name__ == '__main__':
    # load data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    model = Sequential()
    model.add(Reshape((784,), input_shape=(784,)))
    for units in [128, 64, 32, 16, 32, 64, 128]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))

    # train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f'logs/{int(time())}')
    model.fit(mnist.train.images, 
              mnist.train.images,
              validation_data=(mnist.test.images, mnist.test.images),
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE, 
              shuffle=True,
              verbose=1,
              callbacks=[tensorboard])

    # eval
    score = model.evaluate(mnist.test.images, mnist.test.images, batch_size=1)
    print(score)
