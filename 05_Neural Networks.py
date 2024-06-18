# 01. History of Artificial Intelligence
# 02. History of Neural Networks
# 03. Applications of Machine Learning
# 04. Application Areas for Neural Networks
# 05. Motivations to Use Neural Networks
# 06. Neural Networks with Python - TensorFlow and Keras
import tensorflow as tf
import keras

print(tf.constant([1, 2, 3])) # a tensor filled with constants
print(tf.zeros([3, 3])) # a tensor filled with 0

# Computational Graphs
@tf.function
def compute():
    tf1 = tf.constant(2)
    tf2 = tf.constant(3)
    tf3 = tf.constant(5)
    tf_sum = tf.add(tf1, tf2)
    tf_out = tf.multiply(tf3, tf_sum)
    return tf_sum, tf_out

# Using tf.function to create a graph
tf_sum, tf_out = compute()
print(tf_sum)
print(tf_out)

concrete_function = compute.get_concrete_function()
print(concrete_function.graph.get_operations())

# 07. Nueral Networks with Python - Defining a Model
# import
from keras.models import Sequential
from keras.layers import Dense
nn = Sequential() # Initialize a model
# add() adds a new layer to the model
nn.add(Dense(1, input_dim=4, activation='sigmoid'))
nn.summary() # Summary() provides an overview of the model.

# 08. Neural Networks with Python - Compiling and Training a Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ensure TensorFlow 2.x is being used
print("TensorFlow version:", tf.__version__)

# Sample data (ensure to replace this with actual data)
import numpy as np
X_train = np.random.rand(100, 4)  # Example training data
y_train = np.random.randint(2, size=100)  # Example training labels
X_test = np.random.rand(20, 4)  # Example test data
y_test = np.random.randint(2, size=20)  # Example test labels

# Initialize a model
nn = Sequential()
# Add a layer to the model
nn.add(Dense(1, input_dim=4, activation='sigmoid'))
# Compile the model
nn.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])
# Train the model with validation data
history = nn.fit(X_train,
                 y_train,
                 epochs=1000,
                 batch_size=100,
                 validation_data=(X_test, y_test))
# Print the summary of the model
nn.summary()

# 09. Neural Networks with Python - Understanding the Input
# 10. Neural Networks with Python - Understanding the Output
# 11. Neural Networks with Python - What Happens during Learning
# 12. Neural Networks with Python - Evaluating a Trained Model
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 13. Biological Motivation
# 14. Neural Networks Architectures
# 15. Deep Learning
# 16. Model Saving and Loading
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Create the directory if it does not exist
if not os.path.exists('models'):
    os.makedirs('models')
# Save the model architecture to JSON
model_json = nn.to_json()
with open('models/nn_architecture.json', 'w') as json_file:
    json_file.write(model_json)
# Save the model weights to an H5 file
nn.save_weights('models/nn_weights.h5')
# Load the model architecture from JSON
with open('models/nn_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
# Recreate the model from JSON
loaded_model = model_from_json(loaded_model_json)
# Load the model weights
loaded_model.load_weights('models/nn_weights.h5')
# Compile the loaded model (required before making predictions or evaluations)
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
# Verify the loaded model's structure
loaded_model.summary()

# 17. Hyperparameter Tuning
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Function to create the model, required for KerasClassifier
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=11, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Create a KerasClassifier object
kc = KerasClassifier(model=create_model, verbose=0)

# Define the hyperparameters
epochs = [50, 100]
batch_size = [50, 200]

# Create a grid
param_grid = dict(epochs=epochs, batch_size=batch_size)
# Perform a grid search
grid = GridSearchCV(estimator=kc, param_grid=param_grid, n_jobs=-1, cv=3)
# Sample data (ensure to replace this with actual data)
import numpy as np
X_train = np.random.rand(100, 11)  # Example training data with 11 features
y_train = np.random.randint(2, size=100)  # Example binary training labels
# Fit the grid search to the data
grid_result = grid.fit(X_train, y_train)
# Summarize the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 6: Neural Networks
'''

# pip uninstall h5py
# pip install h5py
# pip install keras
# pip install tensorflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json

import joblib
import matplotlib.pyplot as plt
import pathlib
pathlib.Path('models').mkdir(parents=True, exist_ok=True)
pathlib.Path('plots').mkdir(parents=True, exist_ok=True)
###############################################################################

#%% ------------------------------------------------------------------------------------
# BLOCK 1: Data Preparation
# -------------------------
print('#'*50)
print('########## Data Preparation ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Create the 'red' dataset from the following URL:
# 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
print('---------- EX-01 ----------')

red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
red = pd.read_csv(red_url, sep=';')

#%% ------------------------------------------------------------------------------------
# Execute the following commands to set a global seed, namely the same seed as in auxiliary_functions.py to be able to compare the results later.
global_seed = 1418
np.random.seed(global_seed)

#%% ------------------------------------------------------------------------------------
# EX02: Split the dataset into
# - a pandas series 'y' with the binary target variable quality>= 6 or <6 and
# - a pandas data frame 'X' with all other variables
print('---------- EX-02 ----------')

X = red.drop('quality', axis = 1)
y = (red['quality'] >= 6)*1

#%% ------------------------------------------------------------------------------------
# EX03: Perform a train-test split with the ratio of 80:20
print('---------- EX-03 ----------')

# X_train, X_test, y_train, y_test = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#%% ------------------------------------------------------------------------------------
# BLOCK 2: Single Layer Perceptron (SLP)
# --------------------------------------
print('#'*50)
print('########## Single Layer Perceptron (SLP) ##########')
print('#'*50)

# We are now going to build a SLP to predict the wine quality.

#%% ------------------------------------------------------------------------------------
# EX01: initialize a neural network model with keras.models.Sequential()
print('---------- EX-01 ----------')

slp = Sequential()

#%% ------------------------------------------------------------------------------------
# EX02: Add a dense layer with 11 input nodes and 1 output nodes to the SLP.
# Set the activation function of the layer to be 'sigmoid'.
print('---------- EX-02 ----------')

# slp.add(...)
slp.add(Dense(1, input_dim=11, activation='sigmoid'))

#%% ------------------------------------------------------------------------------------
# EX03: Display an overview of the SLP
print('---------- EX-03 ----------')

overview = slp.summary()

#%% ------------------------------------------------------------------------------------
# EX04: Compile the SLP and set following details:
# 1) loss => 'binary_crossentropy'
# 2) optimizer => 'Adam'
# 3) metric => 'Accuracy'
print('---------- EX-04 ----------')

# slp.compile(...)
slp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% ------------------------------------------------------------------------------------
# EX05: # Fit the SLP on the training data.
# You can try different number for epochs and different batch sizes. Try to improve your accuracy by adjusting epochs and batch sizes.
# Note: You have to reload the model each time so that the weights are re-initialized.
# Store the training progress in the history variable.
print('---------- EX-05 ----------')

# history = slp.fit(...)
history = slp.fit(X_train,
                  y_train,
                  epochs=1000,
                  batch_size=100)

#%% ------------------------------------------------------------------------------------
# EX06: Run the following code to fit a logit model on the same training data.
print('---------- EX-06 ----------')

logit_reg = LogisticRegression()
lr = logit_reg.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX07: Use both the SLP and the Logit to predict the test data.
print('---------- EX-07 ----------')

lr_testpred = lr.predict(X_test)
perc_testpred = (slp.predict(X_test) > 0.5).astype("int32")

#%% ------------------------------------------------------------------------------------
# EX08: Compare the accuracy and confusion matrix of the two models via the following code.
# Which model is better?
print('---------- EX-08 ----------')

print('Accuracy Perceptron:', accuracy_score(y_test, perc_testpred))
print('Accuracy Logit:', accuracy_score(y_test, lr_testpred))
print('Confusion Matrix Perceptron :')
print(confusion_matrix(y_test, perc_testpred))
print('Confusion Matrix Logit :')
print(confusion_matrix(y_test, lr_testpred))


#%% ------------------------------------------------------------------------------------
# BLOCK 2: Neural Network
# -----------------------
print('#'*50)
print('########## Neural Network ##########')
print('#'*50)


#%% ------------------------------------------------------------------------------------
# EX01: Initialize a neural network with keras.models.Sequential()
print('---------- EX-01 ----------')

nn = Sequential()

#%% ------------------------------------------------------------------------------------
# EX02: Add nine dense layers to the network with number of nodes: 12, 14, 16, 18, 16, 14, 12, 8, and 1.
# Let the activation function of all layers be 'relu', except for the last one which is 'sigmoid'.
# Note: the number of covariates must be consistant with the 'input_dim' in the first layer.
print('---------- EX-02 ----------')

nn.add(Dense(12, input_dim=11, activation='relu'))
nn.add(Dense(14, activation='relu'))
nn.add(Dense(16, activation='relu'))
nn.add(Dense(18, activation='relu'))
nn.add(Dense(16, activation='relu'))
nn.add(Dense(14, activation='relu'))
nn.add(Dense(12, activation='relu'))
nn.add(Dense(8, activation='relu'))
nn.add(Dense(1, activation='sigmoid'))

#%% ------------------------------------------------------------------------------------
# EX03: Display an overview of the model
print('---------- EX-03 ----------')

overview = nn.summary()

#%% ------------------------------------------------------------------------------------
# EX04: Compile the model and set following details:
# 1) loss => 'binary_crossentropy'
# 2) optimizer => 'Adam'
# 3) metric => 'Accuracy'
print('---------- EX-04 ----------')

# slp.compile(...)
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% ------------------------------------------------------------------------------------
# EX05: # Fit the model on the training data.
# You can try different number for epochs and different batch sizes. Try to improve your accuracy by adjusting epochs and batch sizes.
# Note: You have to reload the model each time so that the weights are re-initialized.
# Store the training progress in the history variable.
print('---------- EX-05 ----------')

history = nn.fit(X_train,
                  y_train,
                  epochs=1000,
                  batch_size=100)


#%% ------------------------------------------------------------------------------------
# EX06: Keep an eye on the following plots to avoid overfitting.
# To see overfitting - increase the number of epochs.
print('---------- EX-06 ----------')

f0 = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
f0.savefig('plots/nn_training_acc.png')

f1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Current Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
f1.savefig('plots/nn_training_loss.png')

#%% ------------------------------------------------------------------------------------
# EX07: Use both the model to predict the test data.
print('---------- EX-07 ----------')

nn_testpred = (nn.predict(X_test) > 0.5).astype("int32")


#%% ------------------------------------------------------------------------------------
# EX08: Compare your new model with the results of the SLP and the Logit model
# with the following code. Which model is better?
print('---------- EX-08 ----------')

print('Accuracy Perceptron:', accuracy_score(y_test, perc_testpred))
print('Accuracy Logit:', accuracy_score(y_test, lr_testpred))
print('Accuracy NN:', accuracy_score(y_test, nn_testpred))

print('Confusion Matrix Perceptron :')
print(confusion_matrix(y_test, perc_testpred))
print('Confusion Matrix Logit :')
print(confusion_matrix(y_test, lr_testpred))
print('Confusion Matrix NN :')
print(confusion_matrix(y_test, nn_testpred))


#%% ------------------------------------------------------------------------------------
# EX09: Save the trained 'nn' model as json & h5 files.
print('---------- EX-09 ----------')

model_json = nn.to_json()
with open('models/nn_architecture.json', 'w') as json_file:
    json_file.write(model_json)
nn.save_weights('models/nn.h5')
print('Saved model to disk')

#%% ------------------------------------------------------------------------------------
# EX10: Load the model from the json and h5 files.
print('---------- EX-10 ----------')

json_file = open('models/nn_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('models/nn.h5')
print('Loaded model from disk')

#%% ------------------------------------------------------------------------------------
# BLOCK 3: Grid Search
# --------------------
print('#'*50)
print('########## Grid Search ##########')
print('#'*50)

# In this exercise, a grid search is to be built to improve the model performance.

#%% ------------------------------------------------------------------------------------
# EX01: Define a function 'create_model()' that builds and returns a NN model sketched above (including the compile step).
print('---------- EX-01 ----------')

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=11, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%% ------------------------------------------------------------------------------------
# EX02: Use the create_model() function to create a KerasClassifier object.
print('---------- EX-02 ----------')

kc = KerasClassifier(build_fn=create_model)

#%% ------------------------------------------------------------------------------------
# EX03: create a dictionary that contains the values to be tested for
# epochs (list of integer values) and batch_size (list of integer values).
print('---------- EX-03 ----------')

param_grid = dict(epochs = [50, 100], batch_size = [50, 200])

#%% ------------------------------------------------------------------------------------
# EX04: create a GridSearch CV object with 3-fold cross-validation;
# fit it on the training data.
print('---------- EX-04 ----------')

grid = GridSearchCV(estimator = kc, param_grid = param_grid, cv = 3)
grid_result = grid.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX05: Use the following codes to output the results of the grid search in the console
print('---------- EX-05 ----------')


print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%s (%s) with: %s' % (mean, stdev, param))


def best_saver(grid_result, X_train, y_train, X_test, y_test, h5_path='models/190904_kc.h5', json_path='models/190904_kc.json'):
    best_grid=grid_result.estimator.build_fn()
    history = best_grid.fit(X_train, y_train, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'],validation_data=(X_test, y_test))
    model_json = best_grid.to_json()
    with open(json_path, 'w') as json_file:
        json_file.write(model_json)
    best_grid.save_weights(h5_path)
    print('Saved the best model on disk.')
def best_loader(h5_path, json_path):
    json_file = open(json_path, 'r')
    nn_gs_json = json_file.read()
    json_file.close()
    nn_gs = model_from_json(nn_gs_json)
    nn_gs.load_weights(h5_path)
    return(nn_gs)

#%% ------------------------------------------------------------------------------------
# EX06: the function best_saver() saves the best estimator found by the grid search algorithm
# The function best_loader() loads the best mode saved by best_saver().
# Using the two functions:
# 1) find the best model on the gird & save its as json and h5 files
# 2) load the saved model
# 3) Use the saved model to make prediction on the test data
print('---------- EX-06 ----------')


best_saver(grid_r
nn_gs= best_loader(h5_path='models/190904_kc.h5', json_path='models/190904_kc.json')
nn_gs_testpred = (nn_gs.predict(X_test) > 0.5).astype("int32")


#%% ------------------------------------------------------------------------------------
# BLOCK 4: Comparison with Previous Models
# ----------------------------------------
print('#'*50)
print('########## Comparison with Previous Models ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: # Load the results of the models from previous Exercise / Binary Classification
print('---------- EX-01 ----------')

results_gbc = joblib.load('models/210805_gbc.pkl')
results_rf = joblib.load('models/210805_rf.pkl')

#%% ------------------------------------------------------------------------------------
# EX02: Use the following codes to compare the preset results with the previous results
print('---------- EX-02 ----------')

print('Accuracy Perceptron:', accuracy_score(y_test, perc_testpred))
print('Accuracy NN:', accuracy_score(y_test, nn_testpred))
print('Accuracy NN_GS:', accuracy_score(y_test, nn_gs_testpred))
print('Accuracy Logit:', accuracy_score(y_test, lr_testpred))
print('Accuracy RF:', results_rf['test_accuracy'])
print('Accuracy GBC:', results_gbc['test_accuracy'])

print('Confusion Matrix Perceptron :')
print(confusion_matrix(y_test, perc_testpred))
print('Confusion Matrix NN :')
print(confusion_matrix(y_test, nn_testpred))
print('Confusion Matrix NN_GS :')
print(confusion_matrix(y_test, nn_gs_testpred))
print('Confusion Matrix Logit :')
print(confusion_matrix(y_test, lr_testpred))
print('Confusion Matrix RF :')
print(results_rf['test_confusion'])
print('Confusion Matrix GBC :')
print(results_gbc['test_confusion'])