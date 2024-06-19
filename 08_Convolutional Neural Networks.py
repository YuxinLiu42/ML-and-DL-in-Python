import pandas as pd
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
import pathlib

# 1. CNNs
# 2. CNNs - Functionality
# 3. CNNs - What do CNNs learn?
# 4. CNNs with Python - Reading Data
# Read all images in a folder

for img in os.listdir(folder):
    cv2.imread(folder+'/'+img)
# read and resize images
for img in os.listdir(folder):
to_append = cv2.imread(folder+'/'+img)
to_append = cv2.resize(to_append,(299,299))
X.append(to_append)
y.append(folder)

# 5. CNNs with Python - Defining a Model
# Create a convolutional neural network
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3),
       activation='relu', input_shape=(500,500,3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(7, activation='softmax'))
cnn.summary()

# 6. CNNs with Python - Compiling and Training a Model
# Compile the model
cnn.compile(optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])
# Train the model
history = cnn.fit(X_train, y_train,
         batch_size=50,
         epochs=10,
         validation_data=(X_test, y_test))

# 7. CNNs with Python - Understanding the Input
keras.layers.Conv2D( # The Keras class for convolutional layers
filters, # The number of filters (int)

kernel_size, # The size of the filter (tuple or int)

strides=(1, 1), # int or tuple of 2 ints
activation=None, # Activation function

padding='valid', # 'valid' (= no padding) or 'same'

keras.layers.MaxPooling2D( # The Keras class for pooling layers

pool_size=(2, 2) # Size of the pooling matrix
strides=None # if None: strides = pool_size
padding='valid', # 'valid' (= no padding) or 'same' ...)

# 8. CNNs with Python - Understanding the Output
# 9. CNNs with Python - What Happens during Learning
# 10. CNNs with Python - Evaluating a Trained Model
# Plot the training process
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Current Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Apply the model to the training data
results = cnn.predict(X_test) # softmax outputs per image

# Consider the accuracy on the validation data
accuracy_score(y_test, results)

# Create a confusion matrix
confusion_matrix(y_test, results)

# 11. Transfer Learning in Python
# Load Inception
model_inception = InceptionResNetV2(include_top=False, pooling='max')
# Push images through inception
predictions = model_inception.predict(img)
# predictions = list of length 1536 with floats


'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 8: Convolutional Neural Networks
'''

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
import pathlib

pathlib.Path('models').mkdir(parents=True, exist_ok=True)
pathlib.Path('plots').mkdir(parents=True, exist_ok=True)
# matplotlib.use("TKAgg")
###############################################################################

# %% ------------------------------------------------------------------------------------
# BLOCK 1: Preprocessing
# ----------------------
print('#' * 50)
print('########## Preprocessing ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Execute the following code to:
# unzip the training_imgs.zip file => save the content in the training_imgs folder
print('---------- EX-01 ----------')

import zipfile

with zipfile.ZipFile('data/training_imgs.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

# %% ------------------------------------------------------------------------------------
# EX02: The images are now in 'data/training_imgs'.
# - read the images one after the other and write them into one array.
# - write the class of the images into a second array (the folder names indicate the classes)
# - when reading the images, resize them to 299 x 299 pixels.
# Hence, you would have:
# - X: NumPy array with images (shape: (number of images, 299, 299, 3))
# - y: NumPy array with the class numbers (shape: (number of images,))
print('---------- EX-02 ----------')

X, y = [], []
classes_count = 0

for folder in os.listdir('./data/training_imgs'):
    if
folder in ['sonnenblume', 'tellerkraeuter', 'sommeraster']:
print(folder)
for img in os.listdir(os.path.join('./data/training_imgs', folder)):
    if
img.endswith('jpg'):
try:
    to_append = cv2.imread(os.path.join('./data/training_imgs', folder, img))
    to_append = cv2.resize(to_append, (299, 299))
    X.append(to_append)
    y.append(classes_count)
except Exception as e:
    print(e)
classes_count += 1

X = np.array(X)
y = np.array(y)

# %% ------------------------------------------------------------------------------------
# EX03: Shuffle X and y together.
# Use sklearn.utils.shuffle() instead of numpy.random.shuffle(),
# because the latter can only handle one argument,
# but here it is important that X and y are shuffled similarly.
# Set a seed beforehand
print('---------- EX-03 ----------')

global_seed = 1420
np.random.seed(global_seed)
X, y = sklearn.utils.shuffle(X, y)

# %% ------------------------------------------------------------------------------------
# EX04: Perform a train-test-split with the test-ratio of 0.2, and stratified on y
print('---------- EX-04 ----------')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y)

# %% ------------------------------------------------------------------------------------
# EX05: One-hot-encode the target variable y.
# This should be done for both y_train and y_test.
print('---------- EX-05 ----------')

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape((-1, 1)))
y_test = ohe.transform(y_test.reshape((-1, 1)))

# %% ------------------------------------------------------------------------------------
# BLOCK 2: Model Building/Training
# --------------------------------
print('#' * 50)
print('########## Model Building/Training ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Use the following architecture as your model, but
# add the input and output dimensions so you can train a complete CNN
print('---------- EX-01 ----------')

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(299, 299, 3)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(32, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(3, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# %% ------------------------------------------------------------------------------------
# EX02: Fit the model on the training data.
# You can choose any number of epochs and batch size and try them out
# It takes longer to train an epoch than in exercise sheet 6.
# Therefore, do not set the epoch size above 15 (to test the code, it is best to set it to 1)
# If time allows: try to improve the accuracy by adjusting epochs and batch sizes.
# Note: You should reload the model each time so that the weights are re-initialized.
# Take X_test and y_test for validation.
# Save the result in the history variable.
print('---------- EX-02 ----------')

history = cnn.fit(X_train, y_train,
                  batch_size=50,
                  epochs=10,
                  verbose=1,
                  validation_data=(X_test, y_test))

# %% ------------------------------------------------------------------------------------
# EX03: Keep an eye on the following plots to avoid overfitting.
print('---------- EX-03 ----------')

f0 = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show(block=False)
plt.close()
f0.savefig('plots/cnn_training_acc.png')

f0 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Current Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show(block=False)
plt.close()
f0.savefig('plots/cnn_training_loss.png')

# %% ------------------------------------------------------------------------------------
# EX04: Use model.evaluate() method to examine the model performance.
print('---------- EX-04 ----------')

performance = cnn.evaluate(X_test, y_test, verbose=0)
print('Test loss:', performance[0])
print('Test accuracy:', performance[1])

# %% ------------------------------------------------------------------------------------
# EX05: Save the model as a json and an h5 file.
print('---------- EX-05 ----------')

model_json = cnn.to_json()
with open('models/cnn_architecture.json', 'w') as json_file:
    json_file.write(model_json)

cnn.save('models/cnn')
print('Saved model to disk')

# %% ------------------------------------------------------------------------------------
# BLOCK 3: Model Prediction
# -------------------------
print('#' * 50)
print('########## Model Prediction ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Load the (trained) model to make a prediction on the validation images.
# Also, inspect the difference between the two save & load functions:
# what information do they store, and what information do they not store?
print('---------- EX-01 ----------')

with open('models/cnn_architecture.json', 'r') as json_file:
    cnn_loaded_json = json_file.read()
    cnn_loaded_from_json = model_from_json(cnn_loaded_json)

cnn_loaded = load_model('models/cnn')
print('Loaded model from disk')

# %% ------------------------------------------------------------------------------------
# EX02: The images are stored in X_test.
# Save the true and predicted values in two lists: y_true, y_pred
# For each image, print the true class, the predicted class, and the prediction confidence
print('---------- EX-02 ----------')

classes = ['sonnenblume', 'tellerkraeuter', 'sommeraster']

y_true, y_pred = [], []
for x_arr, y_arr in zip(X_test, y_test):
    print('-' * 20)
    prediction = cnn_loaded.predict(np.array([x_arr]))
    confidence = np.max(prediction)
    true_class = classes[np.argmax(y_arr)]
    pred_class = classes[np.argmax(prediction)]
    y_true.append(true_class)
    y_pred.append(pred_class)
    print('true_class: {}, pred_class: {}, confidence: {}'.format(true_class, pred_class, confidence))
    print('-' * 20)
    print(' ')

# An alternative way with less details:
# y_true = y_test
# y_pred = cnn_loaded.predict(X_test)

# %% ------------------------------------------------------------------------------------
# EX03: Create a confusion matrix for the predicted values and show it in the console.
print('---------- EX-03 ----------')

confusion = confusion_matrix(y_true, y_pred)
print(confusion)

# %% ------------------------------------------------------------------------------------
# BLOCK 4: Transfer Learning
# --------------------------
print('#' * 50)
print('########## Transfer Learning ##########')
print('#' * 50)

# The required codes for this block are all given => you don't need to change the codes anymore.
# The aim is to understand how transfer learning works.


# Load the Inception model without its fully connected layer.
# All images are predicted once by the Inception model.
model_inception = InceptionResNetV2(include_top=False, pooling='max')


def apply_inception(inp_path, out_path, model):
    classes = os.listdir(inp_path)
    feat = []
    for i in range(len(classes)):
        class_binary = [0] * len(classes)
        class_binary[i] = 1
        images = os.listdir(inp_path + '/' + classes[i])
        for img in images:
            try:
                print(i, img)
                im = image.load_img(inp_path + '/' + classes[i] + '/' + img, target_size=(299, 299))
                x = image.img_to_array(im)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x).tolist()[0]
                ret = [img]
                ret.extend(class_binary)
                ret.extend(preds)
                feat.append(ret)
            except Exception as e:
                print(e)
    feat = pd.DataFrame(feat)
    print(feat.shape)
    feat.to_csv(out_path, sep='|', index=False, header=True)


# The result is a dataset with 1536 columns/features, saved at 'data/cnn_transfer.csv'.
apply_inception('data/training_imgs', 'data/cnn_transfer.csv', model_inception)

# Read the images (that are now translated into tabular data) as a pandas dataframe:
img_flat = pd.read_csv('data/cnn_transfer.csv', sep='|')
img_flat = sklearn.utils.shuffle(img_flat)

# Split the dataset into X and y
# - X: 1536 floats describing each image.
# - y: the one-hot encoded class of the images.
X = img_flat.drop(['0', '1', '2', '3'], axis=1)
y = img_flat[['1', '2', '3']]

# Train-Val-Split in a ratio of 80:20
global_seed = 1420
np.random.seed(global_seed)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y)

# Creation, initialization, compilation, and fitting of the fully connected layers of the CNN:
model_transf = Sequential()
model_transf.add(Dense(int(650), input_dim=1536, activation='relu'))
model_transf.add(Dense(int(400), activation='relu'))
model_transf.add(Dense(int(250), activation='relu'))
model_transf.add(Dense(3, activation='softmax'))
model_transf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_t = model_transf.fit(X_train,
                             y_train,
                             epochs=18,
                             batch_size=20,
                             validation_data=(X_test, y_test))

# Check the training results:
f0 = plt.figure()
plt.plot(history_t.history['accuracy'])
plt.plot(history_t.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show(block=False)
plt.close()
f0.savefig('plots/cnn_training_transfer_acc.png')

f0 = plt.figure()
plt.plot(history_t.history['loss'])
plt.plot(history_t.history['val_loss'])
plt.title('Current Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show(block=False)
plt.close()
f0.savefig('plots/cnn_training_transfer_loss.png')

# Examine the model performance.

score_transf = model_transf.evaluate(X_test, y_test, verbose=0)  # tbd
print('Test loss:', score_transf[0])
print('Test accuracy:', score_transf[1])

# Apply the model and check its accuracy:
pred_classes = np.argmax(model_transf.predict(X_test), axis=-1)
true_classes = []
for label in np.array(y_test):
    true_classes.append(list(label).index(1))

print('Accuracy CNN Transfer-Learning:', accuracy_score(true_classes, pred_classes))
print(confusion_matrix(true_classes, pred_classes))
