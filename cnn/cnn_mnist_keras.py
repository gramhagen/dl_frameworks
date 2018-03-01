"""Keras MNIST Autoencode: https://blog.keras.io/building-autoencoders-in-keras.html"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Upsampling2D
from keras.metrics import sparse_categorical_accuracy
from keras.callbacks import TensorBoard
from time import time
import tensorflow as tf


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
BATCH_SIZE = 100
NUM_EPOCHS = 5
NUM_CLASSES = 10


if __name__ == '__main__':
    # load data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    # build model
    model = Sequential()
    model.add(Reshape((IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), input_shape=(IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS,)))
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.04))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # train
    optimizer = SGD(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                optimizer=optimizer)

    tensorboard = TensorBoard(log_dir=f'logs/{int(time())}')
    model.fit(mnist.train.images, 
              mnist.train.labels, 
              epochs=NUM_EPOCHS, 
              batch_size=BATCH_SIZE, 
              shuffle=True,
              verbose=1,
              callbacks=[tensorboard])

    # eval
    score = model.evaluate(mnist.test.images, mnist.test.labels, batch_size=1)
    print(score)
