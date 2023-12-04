import tensorflow as tf
import time
import numpy as np
from tensorflow import keras

print(tf.__version__)

# Model for
#  y = (2 * x) - 1
#
# using sigmoid
#

# This is adding a Sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Other optimizers can be tried to reduce mean squared loss. e.g. adam.
model.compile(optimizer='sgd', loss='mean_squared_error')

# input is x and expected output is y for training purpose
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# This will repeat 500 times
model.fit(xs, ys, epochs=500)

# avoid cluttering the stdout with training logs
time.sleep(10)

# predict
test_input = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)
test_output_expected = np.array([19.0, 21.0, 23.0, 25.0, 27.0, 29.0])
test_output = model.predict(test_input)

print("INPUT")
print(test_input)
print("OUTPUT")
print(test_output)
print("EXPECTED")
print(test_output_expected)
