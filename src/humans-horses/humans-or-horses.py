# Chapter 2, Humans or Horses using CNN

import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image

dataset_dir = "/Users/zenkiu/dev/ml/datasets/humans-or-horses/"
training_dir = dataset_dir + "training"
validation_dir = dataset_dir + "validation"
test_image = dataset_dir + "predict/horse_or_human_2.png"
test_image2 = dataset_dir + "predict/horse_or_human_2.png"
test_image3 = dataset_dir + "predict/horse_or_human_3.png"

### All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)
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


def humans_or_horses():
    """

    :return:
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        epochs=15,
        validation_data=validation_generator
    )

    print(model.summary())

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

    return None


humans_or_horses()
