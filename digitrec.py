# MNIST Image Classifier
# Author: Matthew Shiel
# Date: 30-11-2018

# Imports
import os.path as path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import save_model

# Library used to prompt user for file
# dialog that requests selection of an existing file.
import tkinter as tk
from tkinter import filedialog

# def getImage() {

# }

# File explorer image selection
while True:

    # Not necessary to explicitly have this line but it's good practice
    root = tk.Tk()
    root.withdraw()

    # Adapted from https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
    file_path = filedialog.askopenfilename()

