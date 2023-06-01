"""
Download dataset from :

https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html

1. Download the dataset folder and create two folder training set and test set
in the parent dataste folder
2. Move 30-40 image from both TB positive and TB Negative folder
in the test set folder
3. The labels of the images will be extracted from the folder name
the image is present in.

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequentials
from tensorflow.keras import layers, models
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout


all_data_dir = "/Users/aditi/Downloads/CNN_Braintumor/archive/dataset"


if __name__ == "__main__":
    # Initialising the CNN
    # (Sequential- Building the model layer by layer)
    classifier = models.Sequential()

#Convolution 
classifier.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Pooling
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

#Add 2nd convolution layer
classifier.add(layers.Conv2D(32, (3, 3), activation="relu"))
classifier.add(layers.MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(layers.Flatten())

#Step 4 - Full connection
classifier.add(layers.Dense(units=128, activation="relu"))
classifier.add(layers.Dense(units=1, activation="sigmoid"))

# Compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Part 2 - Fitting the CNN to the images

# Load Trained model weights
# from keras.models import load_model
    # regressor=load_model('cnn.h5')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory("dataset/trainingset", target_size=(64, 64), batch_size=32, class_mode="binary")

test_set = test_datagen.flow_from_directory("dataset/testset", target_size=(64, 64), batch_size=32, class_mode="binary")

classifier.fit_generator(training_set, steps_per_epoch=5, epochs=30, validation_data=test_set)

classifier.save("cnn.h5")

#Part 3 - Making new predictions

test_image = tf.keras.preprocessing.image.load_img("dataset/single_prediction/Y9.jpg", target_size=(64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
    
#training_set.class_indices
if result[0][0] == 0:
    prediction = "Normal"
if result[0][0] == 1:
    prediction = "Abnormality detected"

