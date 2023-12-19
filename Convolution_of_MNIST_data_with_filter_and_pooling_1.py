import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    # intent is to break out when accuracy reaches a threshold value.
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.97): # 97%
            print("\n Reached 97% accuracy. Stopping the training....\n\n")
            self.model.stop_training = True
    pass

# Data file mnist.npz is publicly available
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "trainingdata/mnist.npz")

(images, labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)
(images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# reshape into 28 by 28 size C style
images = np.reshape(images, (60000, 28, 28, 1), 'C')
images = np.divide(images, np.max(images))

print(f"Maximum pixel value: {np.max(images)}\n")
print(f"Shape of training set: {images.shape}\n")
print(f"Final shape of each image: {images[0].shape}")

# Define the model. We are using 3 by 3 filter for sharpening the edges
# and compression by 2x2.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        
# Instantiate the callback
callbacks = myCallback()

# Training the model
counter = model.fit(images, labels, epochs=10, callbacks=[callbacks])


print(f"Number of epochs consumed in training: {len(counter.epoch)} epochs")

