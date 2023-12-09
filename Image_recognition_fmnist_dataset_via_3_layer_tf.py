import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# This callback gets calls when accuracy reaches certain threshold so that
# we don't end up running all epochs.
class testCallback(tf.keras.callbacks.Callback):
  def on_done(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.85): # We want to try until 90% correctness. 
      print("\nReached 90% accuracy!")
      self.model.stop_training = True

callbacks = testCallback()

# This is public data set of images 28x28 pixel for large numm
# F-MNIST Dataset Statistics
# The Fashion-MNIST (F-MNIST) dataset is a collection of 60,000 training
# images and 10,000 testing images of handwritten fashion items,
# such as clothing, accessories, and footwear. It is a popular benchmark
# dataset for machine learning tasks, particularly image classification.
# Here are some key statistics about the F-MNIST dataset:
#
# Dataset size:
#
# Training images: 60,000
# Testing images: 10,000
# Image size:
#
# Width: 28 pixels
# Height: 28 pixels
# Channels: 1 (grayscale)
# Number of classes: 10
#
# Class labels: 0: T-shirt/top 1: Trouser 2: Pullover 3: Dress 4:
# Coat 5: Sandal 6: Shirt 7: Sneaker 8: Bag 9: Ankle boot

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels) ,  (test_images, test_labels) = fmnist.load_data()

# This part prints the data set image.
index = 0
np.set_printoptions(linewidth=320)
# This is the matrix form of the data.
print(f'LABEL is : {training_labels[index]}')
print(f'\nMATRIX:\n {training_images[index]}')
# This is the visual part of the image.
plt.imshow(training_images[index])

# normalize for pixcel code width
training_images=training_images/255.0
test_images=test_images/255.0


# First layer uses Flatten function because we have 28x28 size vector for each
# input sample. Flattening reduces is to 784x1 instead of having to define 28
# layers of 28 neurons each.
#
# We use activation relu for second layer. It adds non linearity. i.e. the
# output is 0 for negative input and same as input for positive values.
# f(x) = max(0, x)
#
# We use activation softmax for final layer. It bounds the output to 1.
# The softmax function essentially takes a vector of real numbers as input
# and outputs a vector of probabilities, where each element represents the
# probability of the input belonging to a specific class.
# 
# Third (final) layer matches the expected total codes i.e. 10. This means that
# if an input is supplied to this model the final outcome will be one of the 10
# codes. Each code represents a type of image - The class labels explained
# above in F-MNIST dataset. e.g. 2 for Trouser.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(f'\nCLASSIFICATIONS:\n {classifications[0]}')
print(f'\nPREDICTION IS : \n{test_labels[0]}')
print(f'\nDONE\n\n')
