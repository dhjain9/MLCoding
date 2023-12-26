import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True
        

base_dir = "./data/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")

image_data_gen = ImageDataGenerator(rescale=1./125)

# each image is 150x150 and two labels are kept in sub directories under the base_dir.
# the labels are "happy" and "sad". Batch size is randomly selected to be 10.
# We will use binary_crossentropy loss function later matching to the class mode "binary" here.
image_data_gen = image_data_gen.flow_from_directory(directory=base_dir,
                                                    target_size=(150, 150),
                                                    batch_size=10,
                                                    class_mode="binary")
# 3 Convolutional layers with 3 by 3 filter in each later followed by the 2 by 2 pooling. We flatten the
# resuling 16 by 16 image. Then dense layer of 512 neurons followed by binary label at the last layer.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# learning_rate 0.001
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=0.001),
            metrics=['accuracy'])     


trained = model.fit(x=image_data_gen,
                    epochs=20,
                    callbacks=[myCallback()]
                    )

print(f"Reached 99.5 accuracy after {len(trained.epoch)} epochs")

   

    
