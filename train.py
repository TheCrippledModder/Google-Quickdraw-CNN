import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Activation
import matplotlib.pyplot as plt
import numpy as np
import ndjson
import random
import time
import cv2
import os

training_data = []
loss_function = ''

#Variables:
dataset_dir = '' # Should point to ../tinyquickdraw/quickdraw_simplified
img_size = 100 # Image size - default: 100
num_classes = 5 # Number of classes to process - default: 5 (Put None to process all 300+ classes - WARNING: HIGH RAM USAGE) 
img_per_class = 1000 # Number of images per class to process - default: 1000
num_epochs = 10 # The number of epochs to train our model - default: 10
batch_size = 32 # Number of images to be processed by network per step - default: 32
percentage_split = 0.3 # Percentage of your training images that will go to validating the model - default: 0.3


def dump_images():
    for data in os.listdir(dataset_dir)[:num_classes]: 
        with open(os.path.join(dataset_dir, data)) as FILE:
            JSON = ndjson.load(FILE) # Open up JSON file

            counter = img_per_class - 1
            for sketch_data in JSON:
                label = sketch_data['word'].replace(' ', '-')
                drawing = sketch_data['drawing']
                recognized = sketch_data['recognized']
         
                if (recognized == True and counter >=0): # Only dump recognized images
                    counter -= 1
                    for stroke in drawing:
                        plt.plot(stroke[0], stroke[1]) # Plot X and Y for each stroke then combine them together
                        
                    if not os.path.isdir(os.path.join(os.getcwd(), 'train')):
                        os.mkdir('train')

                    if not os.path.isdir(os.path.join(os.getcwd(), 'train', label)):
                        os.mkdir('train//' + label)
                    
                    plt.axis('off') # Turn off axis
                    plt.savefig(os.path.join(os.getcwd(), 'train', label, str(time.time())) + '.png', bbox_inches='tight', pad_inches=0) # Save images 
                    plt.close()

def preprocess_images():
    X = []
    Y = []
    Categories = []

    for category in os.listdir(os.path.join(os.getcwd(), 'train')):
        if category not in Categories:
            Categories.append(category)

        for image in os.listdir(os.path.join(os.getcwd(), 'train', category)):
            try:
                img_array = cv2.imread(os.path.join(os.getcwd(), 'train', category, image))
                resized_img = cv2.resize(img_array, (img_size, img_size)) # Resize the images because full res pictures would be too expensive
                training_data.append([resized_img, Categories.index(category)])
            except Exception as e:
                print(e)

    random.shuffle(training_data) # Always shuffle data so that we do not train on homogeneous data
 
    for samples, labels in training_data:
        X.append(samples)
        Y.append(labels)

    np.save('samples', np.array(X)) # Save samples as numpy arrays 
    np.save('labels', np.array(Y)) # Save labels as numpy arrays

# One hot encode our labels for each sample
# Ex: If we had 10 classes and wanted to encode 5 it would encode to -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def to_categorical(array):
	one_hot = np.zeros((len(array), num_classes))

	for idx, num in enumerate(array):
		one_hot[idx][num] = 1
	return one_hot


def train():
    samples = np.load('samples.npy')
    samples = samples.astype('float32') / 255 # Normalize our data

    if num_classes > 2:
        labels = to_categorical(np.load('labels.npy'))
        loss_function = 'categorical_crossentropy'
    else:
        labels = np.load('labels.npy')
        loss_function = 'binary_crossentropy'

    network = buildModel()
    network.compile(loss=loss_function, optimizer="rmsprop", metrics=['accuracy'])
    network.fit(samples, labels, validation_split=percentage_split, epochs=num_epochs, batch_size=batch_size)

    network.save('Network.model')

    
def buildModel():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
        
    model.add(Flatten())

    if (num_classes > 2):
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
            
    return model

if (os.path.isfile('samples.npy') and os.path.isfile('labels.npy')):
    train()
else:
    dump_images()
    preprocess_images()
    train()
