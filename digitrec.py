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
import tensorflow as tf
from tensorflow.python.keras.models import save_model

# Store 28x28 images of handwritten digits 0-9 from MNIST dataset
mnist = tf.keras.datasets.mnist 

# Load the dataset in to memory
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plot digit to make sure everything is working
plt.imshow(x_train[0])
plt.show()
# File explorer image selection
# while True:

#     # Not necessary to explicitly have this line but it's good practice
#     root = tk.Tk()
#     root.withdraw()

#     # Adapted from https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
#     file_path = filedialog.askopenfilename()

