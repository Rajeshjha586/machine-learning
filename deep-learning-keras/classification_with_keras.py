# ============================================================
# MNIST Handwritten Digit Classification using Keras
# ============================================================

# ------------------------------------------------------------
# Install required libraries
# ------------------------------------------------------------

# TensorFlow/Keras -> Deep Learning framework
# Matplotlib       -> Used for displaying images

# !pip install tensorflow_cpu==2.18.0
# !pip install matplotlib==3.9.2


# ------------------------------------------------------------
# TensorFlow Environment Settings
# ------------------------------------------------------------

import os

# Disable oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hide unnecessary TensorFlow logs
# 0 = all logs
# 1 = filter INFO logs
# 2 = filter WARNING logs
# 3 = filter ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import keras

# Sequential -> Used to create layer-by-layer neural networks
from keras.models import Sequential

# Dense -> Fully connected neural network layer
from keras.layers import Dense

# Input -> Defines input shape
from keras.layers import Input

# Converts labels into one-hot encoded vectors
from keras.utils import to_categorical

# Used to display images
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Import MNIST Dataset
# ------------------------------------------------------------

# MNIST dataset contains:
# - Handwritten digit images (0–9)
# - 28x28 grayscale images

from keras.datasets import mnist


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

# X_train -> training images
# y_train -> training labels
# X_test  -> testing images
# y_test  -> testing labels

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# ------------------------------------------------------------
# Dataset Shape
# ------------------------------------------------------------

# Shape:
# (60000, 28, 28)
#
# Meaning:
# - 60000 training images
# - each image size = 28x28 pixels

X_train.shape


# ------------------------------------------------------------
# Display First Image
# ------------------------------------------------------------

# Displays first handwritten digit image

plt.imshow(X_train[0])


# ------------------------------------------------------------
# Flatten Images
# ------------------------------------------------------------

# Neural networks expect 1D vectors as input
#
# Current image shape:
# 28 x 28
#
# Flattened shape:
# 784

num_pixels = X_train.shape[1] * X_train.shape[2]


# Convert:
# (60000, 28, 28)
#
# into:
# (60000, 784)

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# ------------------------------------------------------------
# Normalize Pixel Values
# ------------------------------------------------------------

# Original pixel range:
# 0 → 255
#
# After normalization:
# 0 → 1
#
# Normalization improves training performance

X_train = X_train / 255
X_test = X_test / 255


# ------------------------------------------------------------
# One-Hot Encode Labels
# ------------------------------------------------------------

# Example:
#
# 5 becomes:
# [0,0,0,0,0,1,0,0,0,0]
#
# Required for multi-class classification

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ------------------------------------------------------------
# Number of Output Classes
# ------------------------------------------------------------

# Since digits are:
# 0 → 9
#
# Total classes = 10

num_classes = y_test.shape[1]

print(num_classes)


# ============================================================
# Build Neural Network Model
# ============================================================

def classification_model():

    # --------------------------------------------------------
    # Create Sequential Neural Network
    # --------------------------------------------------------

    # Architecture:
    #
    # Input Layer
    # → Hidden Layer 1
    # → Hidden Layer 2
    # → Output Layer

    model = Sequential()


    # --------------------------------------------------------
    # Input Layer
    # --------------------------------------------------------

    # Input shape = 784
    # because each image is flattened into 784 values

    model.add(Input(shape=(num_pixels,)))


    # --------------------------------------------------------
    # Hidden Layer 1
    # --------------------------------------------------------

    # Dense layer with:
    # - 784 neurons
    # - ReLU activation function
    #
    # ReLU(x) = max(0, x)

    model.add(Dense(num_pixels, activation='relu'))


    # --------------------------------------------------------
    # Hidden Layer 2
    # --------------------------------------------------------

    # Dense layer with:
    # - 100 neurons
    # - ReLU activation

    model.add(Dense(100, activation='relu'))


    # --------------------------------------------------------
    # Output Layer
    # --------------------------------------------------------

    # Output layer contains 10 neurons
    # because there are 10 digit classes
    #
    # Softmax converts outputs into probabilities

    model.add(Dense(num_classes, activation='softmax'))


    # --------------------------------------------------------
    # Compile Model
    # --------------------------------------------------------

    # Optimizer:
    # Adam -> updates weights efficiently
    #
    # Loss Function:
    # categorical_crossentropy
    #
    # Used for multi-class classification
    #
    # Metric:
    # accuracy

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------
# Build Model
# ------------------------------------------------------------

model = classification_model()


# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

# Training Process:
#
# Forward Propagation
# → Loss Calculation
# → Backpropagation
# → Weight Updates
#
# epochs=10
# Means:
# model sees full dataset 10 times

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    verbose=2
)


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

# Returns:
# - loss
# - accuracy

scores = model.evaluate(X_test, y_test, verbose=0)


# ------------------------------------------------------------
# Print Accuracy
# ------------------------------------------------------------

print(
    'Accuracy: {}% \n Error: {}'.format(
        scores[1],
        1 - scores[1]
    )
)


# ============================================================
# Save Model
# ============================================================

# Saves:
# - model architecture
# - weights
# - optimizer state

model.save('classification_model.keras')


# ============================================================
# Load Saved Model
# ============================================================

pretrained_model = keras.saving.load_model(
    'classification_model.keras'
)


# ============================================================
# Create Deeper Neural Network
# ============================================================

# This model contains:
#
# Input Layer
# → Hidden Layer 1
# → Hidden Layer 2
# → Hidden Layer 3
# → Hidden Layer 4
# → Hidden Layer 5
# → Output Layer

def classification_model_6layers():

    model = Sequential()


    # Input Layer
    model.add(Input(shape=(num_pixels,)))


    # Hidden Layer 1
    model.add(Dense(num_pixels, activation='relu'))


    # Hidden Layer 2
    model.add(Dense(100, activation='relu'))


    # Hidden Layer 3
    model.add(Dense(100, activation='relu'))


    # Hidden Layer 4
    model.add(Dense(100, activation='relu'))


    # Hidden Layer 5
    model.add(Dense(100, activation='relu'))


    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))


    # Compile Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------------------
# Build Deep Model
# ------------------------------------------------------------

model_6layers = classification_model_6layers()


# ------------------------------------------------------------
# Train Deep Model
# ------------------------------------------------------------

model_6layers.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    verbose=2
)


# ------------------------------------------------------------
# Evaluate Deep Model
# ------------------------------------------------------------

scores_6layers = model_6layers.evaluate(
    X_test,
    y_test,
    verbose=0
)


# ------------------------------------------------------------
# Compare Accuracies
# ------------------------------------------------------------

print(
    'Accuracy_3_layers: {}% \n Accuracy_6_layers: {}'.format(
        scores[1],
        scores_6layers[1]
    )
)


# ============================================================
# Continue Training Saved Model
# ============================================================

# Load previously trained model

pretrained_model = keras.saving.load_model(
    'classification_model.keras'
)

print("Pre-trained model loaded successufully")


# ------------------------------------------------------------
# Further Train Model
# ------------------------------------------------------------

# Continue training for 10 more epochs
#
# Total training:
# Previous 10 epochs
# + Current 10 epochs
# = 20 epochs

pretrained_model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    verbose=2
)


# ------------------------------------------------------------
# Evaluate Further-Trained Model
# ------------------------------------------------------------

scores_20_epochs = pretrained_model.evaluate(
    X_test,
    y_test,
    verbose=0
)


# ------------------------------------------------------------
# Compare Accuracy After More Training
# ------------------------------------------------------------

print(
    'Accuracy_10_epochs: {}% \n Accuracy_20_epochs: {}'.format(
        scores[1],
        scores_20_epochs[1]
    )
)