# Chapter 2, Transfer learning
"""
 We’ll use version 3 of the popular Incep‐ tion model from Google,
 which is trained on more than a million images from
  a data‐ base called ImageNet

 Before the model can be used to recognize images,
 it must be trained using a large set of labeled images.
 ImageNet is a common dataset to use.
"""

import urllib.request

import tensorflow.keras.applications.inception_v3
import tensorflow as tf
from keras.optimizers import RMSprop
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image

dataset_dir = "/Users/zenkiu/dev/ml/datasets/humans-or-horses/"
training_dir = dataset_dir + "training"
validation_dir = dataset_dir + "validation"
test_image = dataset_dir + "predict/horse_or_human_2.png"
test_image2 = dataset_dir + "predict/horse_or_human_2.png"
test_image3 = dataset_dir + "predict/horse_or_human_3.png"

### All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

###validation
validation_generator = ImageDataGenerator(rescale=1 / 255)
validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)


weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = tensorflow.keras.applications.inception_v3.InceptionV3(input_shape=(300, 300, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(weights_file)

pre_trained_model.summary()
#Be warned, it's huge. We'll use mixed7 because its output is nice and small -- 7 x 7 images


for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

#Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)


img = image.load_img(test_image, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

image_tensor = np.vstack([x])
classes = model.predict(image_tensor)
print(classes)
print(classes[0])
if classes[0] > 0.5:
    print(test_image + " is a human")
else:
    print(test_image + " is a horse")


