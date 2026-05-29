# ============================================================
# Convolutional Neural Network (CNN) using Keras on MNIST
# ============================================================

# ------------------------------------------------------------
# Install Required Libraries
# ------------------------------------------------------------

# NumPy         -> Numerical operations
# Pandas        -> Data handling (not used directly here)
# TensorFlow    -> Deep learning backend
# Matplotlib    -> Visualization library

# !pip install numpy==2.0.2
# !pip install pandas==2.2.2
# !pip install tensorflow_cpu==2.18.0
# !pip install matplotlib==3.9.2


# ============================================================
# TensorFlow Environment Configuration
# ============================================================

import os

# Disable oneDNN optimization warnings/logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Hide unnecessary TensorFlow logs
# 0 = all logs
# 1 = filter INFO logs
# 2 = filter WARNING logs
# 3 = filter ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================
# Import Keras Libraries
# ============================================================

import keras

# Sequential -> Used to create layer-by-layer neural networks
from keras.models import Sequential

# Dense -> Fully connected neural network layer
from keras.layers import Dense

# Input -> Defines input shape
from keras.layers import Input

# Converts labels into one-hot encoded vectors
from keras.utils import to_categorical


# ------------------------------------------------------------
# CNN Specific Layers
# ------------------------------------------------------------

# Conv2D -> Convolution layer used for feature extraction
from keras.layers import Conv2D

# MaxPooling2D -> Reduces image size while keeping important features
from keras.layers import MaxPooling2D

# Flatten -> Converts multidimensional output into 1D vector
from keras.layers import Flatten


# ============================================================
# Load MNIST Dataset
# ============================================================

# MNIST contains:
# - Handwritten digit images (0–9)
# - 28x28 grayscale images

from keras.datasets import mnist


# ------------------------------------------------------------
# Load Training and Testing Data
# ------------------------------------------------------------

# X_train -> training images
# y_train -> training labels
# X_test  -> testing images
# y_test  -> testing labels

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# ============================================================
# Reshape Images
# ============================================================

# CNN expects input shape:
#
# (samples, height, width, channels)
#
# Original shape:
# (60000, 28, 28)
#
# New shape:
# (60000, 28, 28, 1)
#
# 1 = grayscale image channel

X_train = X_train.reshape(
    X_train.shape[0],
    28,
    28,
    1
).astype('float32')

X_test = X_test.reshape(
    X_test.shape[0],
    28,
    28,
    1
).astype('float32')


# ============================================================
# Normalize Pixel Values
# ============================================================

# Original pixel values:
# 0 → 255
#
# After normalization:
# 0 → 1
#
# This improves model training stability

X_train = X_train / 255
X_test = X_test / 255


# ============================================================
# One-Hot Encode Labels
# ============================================================

# Example:
#
# Digit 5 becomes:
# [0,0,0,0,0,1,0,0,0,0]
#
# Required for multi-class classification

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ------------------------------------------------------------
# Total Number of Output Classes
# ------------------------------------------------------------

# Digits:
# 0 → 9
#
# Total classes = 10

num_classes = y_test.shape[1]


# ============================================================
# CNN Model with One Convolution + Pooling Layer
# ============================================================

def convolutional_model():

    # --------------------------------------------------------
    # Create Sequential CNN Model
    # --------------------------------------------------------

    model = Sequential()


    # --------------------------------------------------------
    # Input Layer
    # --------------------------------------------------------

    # Input image size:
    # 28x28
    #
    # 1 = grayscale channel

    model.add(Input(shape=(28, 28, 1)))


    # --------------------------------------------------------
    # Convolution Layer
    # --------------------------------------------------------

    # Conv2D Parameters:
    #
    # 16        -> Number of filters
    # (5,5)     -> Filter/kernel size
    # strides   -> Filter movement step
    # relu      -> Activation function
    #
    # Purpose:
    # Detect image features like:
    # edges, curves, shapes

    model.add(
        Conv2D(
            16,
            (5, 5),
            strides=(1, 1),
            activation='relu'
        )
    )


    # --------------------------------------------------------
    # Pooling Layer
    # --------------------------------------------------------

    # Reduces image dimensions
    #
    # Benefits:
    # - Faster computation
    # - Less memory usage
    # - Reduces overfitting

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )


    # --------------------------------------------------------
    # Flatten Layer
    # --------------------------------------------------------

    # Converts 2D feature maps into 1D vector
    #
    # Required before Dense layers

    model.add(Flatten())


    # --------------------------------------------------------
    # Fully Connected Hidden Layer
    # --------------------------------------------------------

    # 100 neurons
    # ReLU activation

    model.add(Dense(100, activation='relu'))


    # --------------------------------------------------------
    # Output Layer
    # --------------------------------------------------------

    # 10 neurons because:
    # digits = 0 → 9
    #
    # Softmax converts outputs into probabilities

    model.add(Dense(num_classes, activation='softmax'))


    # --------------------------------------------------------
    # Compile Model
    # --------------------------------------------------------

    # optimizer='adam'
    # -> updates weights efficiently
    #
    # loss='categorical_crossentropy'
    # -> used for multi-class classification
    #
    # metrics=['accuracy']
    # -> tracks prediction accuracy

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Build CNN Model
# ============================================================

model = convolutional_model()


# ============================================================
# Train CNN Model
# ============================================================

# epochs=10
# -> dataset is trained 10 times
#
# batch_size=200
# -> model processes 200 images at one time
#
# validation_data
# -> evaluates performance on test data after each epoch

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=2
)


# ============================================================
# Evaluate CNN Model
# ============================================================

# Returns:
# scores[0] -> loss
# scores[1] -> accuracy

scores = model.evaluate(X_test, y_test, verbose=0)


# ------------------------------------------------------------
# Print Accuracy and Error
# ------------------------------------------------------------

print(
    "Accuracy: {} \n Error: {}".format(
        scores[1],
        100 - scores[1] * 100
    )
)


# ============================================================
# CNN with Two Convolution + Pooling Layers
# ============================================================

# Deeper CNN can learn more complex image features

def convolutional_model():

    model = Sequential()


    # --------------------------------------------------------
    # Input Layer
    # --------------------------------------------------------

    model.add(Input(shape=(28, 28, 1)))


    # --------------------------------------------------------
    # First Convolution Layer
    # --------------------------------------------------------

    model.add(
        Conv2D(
            16,
            (5, 5),
            activation='relu'
        )
    )


    # --------------------------------------------------------
    # First Pooling Layer
    # --------------------------------------------------------

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )


    # --------------------------------------------------------
    # Second Convolution Layer
    # --------------------------------------------------------

    # 8 filters
    # 2x2 kernel size

    model.add(
        Conv2D(
            8,
            (2, 2),
            activation='relu'
        )
    )


    # --------------------------------------------------------
    # Second Pooling Layer
    # --------------------------------------------------------

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )


    # --------------------------------------------------------
    # Flatten Layer
    # --------------------------------------------------------

    model.add(Flatten())


    # --------------------------------------------------------
    # Fully Connected Layer
    # --------------------------------------------------------

    model.add(Dense(100, activation='relu'))


    # --------------------------------------------------------
    # Output Layer
    # --------------------------------------------------------

    model.add(Dense(num_classes, activation='softmax'))


    # --------------------------------------------------------
    # Compile CNN Model
    # --------------------------------------------------------

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Train Deep CNN
# ============================================================

model = convolutional_model()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=2
)


# ============================================================
# Evaluate Deep CNN
# ============================================================

scores = model.evaluate(X_test, y_test, verbose=0)

print(
    "Accuracy: {} \n Error: {}".format(
        scores[1],
        100 - scores[1] * 100
    )
)


# ============================================================
# Experiment: Effect of Batch Size
# ============================================================

# Larger batch size:
# - faster training
# - uses more memory
# - may slightly affect accuracy

model = convolutional_model()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=1024,
    verbose=2
)


# ------------------------------------------------------------
# Evaluate Model
# ------------------------------------------------------------

scores = model.evaluate(X_test, y_test, verbose=0)

print(
    "Accuracy: {} \n Error: {}".format(
        scores[1],
        100 - scores[1] * 100
    )
)


# ============================================================
# Experiment: Effect of More Epochs
# ============================================================

# epochs=25
# means:
# model sees full dataset 25 times
#
# More epochs:
# - may improve accuracy
# - may increase overfitting
# - increases training time

model = convolutional_model()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=1024,
    verbose=2
)


# ------------------------------------------------------------
# Final Evaluation
# ------------------------------------------------------------

scores = model.evaluate(X_test, y_test, verbose=0)

print(
    "Accuracy: {} \n Error: {}".format(
        scores[1],
        100 - scores[1] * 100
    )
)