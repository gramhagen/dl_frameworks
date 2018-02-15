"""PyTorch MNIST Classifier: https://github.com/pytorch/examples/blob/master/mnist/main.py
other examples: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
"""

from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import numpy as np


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNELS = 1
BATCH_SIZE = 100
NUM_EPOCHS = 5
NUM_CLASSES = 10


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(7*7*64, 1024),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        out = x.view(x.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out


if __name__ == '__main__':

    # load the data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_dataset = TensorDataset(torch.from_numpy(mnist.train.images),
                                  torch.from_numpy(mnist.train.labels))
    test_dataset = TensorDataset(torch.from_numpy(mnist.test.images),
                                 torch.from_numpy(mnist.test.labels))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)

    # create logs for tensorboard
    logger = tf.summary.FileWriter(f'./logs/{int(time())}', flush_secs=1)

    model = CNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = Variable(data), Variable(labels)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.data.max(1, keepdim=True)[1]
            accuracy = pred.eq(labels.data.view_as(pred)).float().mean()

            if step % 100 == 0:
                print(f'Train Epoch: {epoch}\tBatch: {batch_idx}\tLoss: {loss.data[0]}')
                info = dict(loss=loss.data[0], accuracy=accuracy)
                for tag, value in info.items():
                    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                    logger.add_summary(summary, step)
            step += 1
    
    logger.close()

    model.eval()
    test_loss = 0
    test_accuracy = 0
    for data, labels in test_dataloader:
        data, labels = Variable(data), Variable(labels)
        outputs = model(data)
        test_loss += loss_fn(outputs, labels).data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(labels.data.view_as(pred)).sum()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy /= len(test_dataloader.dataset)
    print(dict(loss=test_loss, accuracy=test_accuracy))