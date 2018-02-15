<h1>TensorFlow</h1>
<h2>Install:</h2>

Reference: https://www.tensorflow.org/install/

```bash
brew upgrade python3
mkvirtualenv ~/venvs/tensorflow -p `which python3`
workon tensorflow
pip install --upgrade tensorflow
# version 1.5.0
pip install ipython numpy
```

<h2>Validate:</h2>

Reference: https://www.tensorflow.org/install/install_mac#ValidateYourInstallation

(in python)
```python
import tensorflow as tf
hello = tf.constant('Hello!')
sess = tf.Session()
print(sess.run(hello))
```

<h2>Execute:</h2>

```bash
python cnn_mnist_tf.py &
tensorboard --logdir=./models/mnist_convnet_model &
open http://localhost:6006
```

<h1>Keras</h1>
<h2>Install:</h2>

Reference: https://keras.io/#installation

```bash
workon tensorflow
pip install keras
# version 2.1.4
```

<h2>Execute:</h2>

``` bash
python cnn_mnist_keras.py &
tensorboard --logdir=logs &
open http://localhost:6006
```


<h1>PyTorch</h1>

<h2>Install:</h2>

Reference: http://pytorch.org/

```bash
workon tensorflow
pip install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl 
pip install torchvision 
```

<h2>Execute:</h2>

``` bash
python cnn_mnist_pytorch.py &
tensorboard --logdir=logs &
open http://localhost:6006
```
