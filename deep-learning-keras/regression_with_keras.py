# !pip install numpy==2.0.2
# !pip install pandas==2.2.2
# !pip install tensorflow_cpu==2.18.0

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

import warnings
warnings.simplefilter('ignore', FutureWarning)

filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

concrete_data.head()

print(concrete_data.shape)

concrete_data.describe()

concrete_data.isnull().sum()

# Split data into predictors and target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

predictors.head()

target.head()

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Now using the same dateset,try to recreate regression model featuring five hidden layers,
# each with 50 nodes and ReLU activation functions, a single output layer, optimized using the Adam optimizer.

def regression_model():
    input_colm = predictors_norm.shape[1]  # Number of input features
    # create model
    model = Sequential()
    model.add(Input(shape=(input_colm,)))  # Set the number of input features
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Output layer

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate the model simultaneously using the fit() method
# by reserving 10% of the data for validation and training the model for 100 epochs

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)


# Based on the results, we notice that:
#
# 1. Adding more hidden layers to the model increases its capacity to learn and represent complex relationships within the data.
#    This allows the model to better identify, as a result, the model becomes more effective at fitting the training data and potentially improving its predictions.
# 2. By reducing the proportion of data set aside for validation and using a larger portion for training,
#    the model has access to more examples to learn from. This additional training data helps
#    the model improve its understanding of the underlying trends, which can lead to better overall performance.



