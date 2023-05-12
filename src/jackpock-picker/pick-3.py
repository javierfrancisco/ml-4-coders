import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    model = Sequential([Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)
    ys = np.array([5, 45, 447, 168, 25, 249, 345, 278, 249, 359, 178, 15, 35], dtype=float)
    model.fit(xs, ys, epochs=5000)
    print(model.predict([14.0]))
    #18.976208

# See PyCharm help at https://www.jetbrains.com/help/pycharm/