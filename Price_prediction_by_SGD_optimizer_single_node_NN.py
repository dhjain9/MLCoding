import tensorflow as tf
import numpy as np

def predict_price(optimizer, loss):
    # Base price 0.5. Then for each item 0.5
    training_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    training_output = np.array([1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)

    # Single node.
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(training_input, training_output, epochs=500)
    return model

# Stochastic Gradient Descent
model = predict_price(optimizer='sgd', loss='mean_squared_error')

test_input = 8.0
value = model.predict([test_input])[0]
print(value)
    
