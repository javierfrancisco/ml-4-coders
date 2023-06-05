# Using Public Datasets with TensorFlow Datasets
# TensorFlow datasets is separate from Tensorflow, hence needs to be installed

import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")
for item in mnist_data:
    print(item)

# mnist_data is a dictionary containing two strings, 'test' and 'train'.
# these are the available splits, now let's use a split

mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

# The output is a PrefetchDataset, which can be used to iterate through to inspect the data.
# One nice feature of this adapter is that you can simply call `take(1)` to get the first record.

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())

# The output of the first print is a dictionary, meaning each record is a dictionary.
# When printing the keys, the types are image and label.
# let's print the values

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])

# the output for the image is a 28 x 28 array of values (in a tf.Tensor) from
# 0-255 representing the pixel intensity. The label will be output as
# tf.Tensor(2, shape=(), dtype=int64), indicating that this image is class 2 in the dataset

# let's print data about the dataset

mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
print(info)

# in chapter 2:
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels),
# (test_images, test_labels) = mnist.load_data()

# now with TFDS is little different, it has to convert to numpy array

(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('fashion_mnist',
                                                     split=['train', 'test'],
                                                     batch_size=-1,
                                                     as_supervised=True))
