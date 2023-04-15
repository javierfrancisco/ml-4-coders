# Chapter 2, Introduction to Computer Vision

import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\Reached 96% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

def computer_vision_example():
    """
    Creates a neural network to recognize items of clothing, using a well-known dataset
    called Fashion MNIST. The images are 28 x 28 grayscale. This dataset is comprised of
    images of 70,000 handwritten digits from 0 to 9.
    :return:
    """
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50, callbacks=callbacks)

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[4])
    print(test_labels[4])

    return None


computer_vision_example()



