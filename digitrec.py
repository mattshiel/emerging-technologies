# MNIST Image Classifier
# Author: Matthew Shiel
# Date: 30-11-2018

# Imports
import os.path as path

# Library used to prompt user for file
# dialog that requests selection of an existing file.
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import save_model

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

MODEL_NAME = "digitClassifier.h5"

def prepDataset():
    # Store 28x28 images of handwritten digits 0-9 from MNIST dataset
    mnist = tf.keras.datasets.mnist 

    # Load the dataset in to memory
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    # In this case rescaling the data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    return x_train, y_train, x_test, y_test

def loadClassifier(MNIST_data):
    
    x_train, y_train, x_test, y_test = MNIST_data

    #2 Types of models, Sequential is the most common
    model = tf.keras.models.Sequential()

    # Often times at the end of a CNN there will be a densely connected layer which needs to be flattened. It gets the output of the 
    # convolutional layers, flattens all of its structure to create a single long feature vector to be used by the 
    # dense layer for the final classification.
    model.add(tf.keras.layers.Flatten())

    # Activation is what makes the neuron fire
    # tf.nn.relu is a very standard default activation function, it can be tweaked to possibly improve performance
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # This layer is the output layer, the output layer will always have the number of classifications, 10 in this case
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # Loss is, in simple terms, the degree of error
    # Neural Networks aren't necessarily trying to maximise efficiency but minimize loss
    model.compile(optimizer ='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    #Train the model
    #An epoch is a full pass through the entire dataset
    model.fit(x_train, y_train, epochs=3)

    # Calculate and display the loss and model accuracy 
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    # Save the model
    model.save(MODEL_NAME)
    print('Saved model as ' + MODEL_NAME)


def makePrediction(MNIST_data):

    # Get the data necessary to test
    x_train, y_train, x_test, y_test = MNIST_data

    # Load the model
    active_model = tf.keras.models.load_model(MODEL_NAME)

    # Generate a random number between 1-10000
    randTestImg = np.random.randint(1,10000)

    # Make predictions 
    predictions = active_model.predict(x_test)

    # Print out prediction
    print('The model predicts this is a', np.argmax(predictions[randTestImg]))

    # Show what the number actually was
    print('The image was actually: ')
    plt.imshow(x_test[randTestImg])
    plt.show()

def userMenu():
    ans=True
    while ans:
        print ("""
        ======================
        MNIST IMAGE CLASSIFIER
        ======================
        1. Create and Load Basic Classifier
        2. Make prediction
        3.Exit

        """)
        ans=input("Please make a selection: ")

        MNIST_data = prepDataset()
 
        if ans=="1": 
            loadClassifier(MNIST_data)
        elif ans=="2":
            makePrediction(MNIST_data)
            exit()
        elif ans !="":
            print("\n Not Valid Choice Try again") 


userMenu()
