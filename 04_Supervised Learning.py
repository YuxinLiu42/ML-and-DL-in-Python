# 0. Machine Learning
# Supervised Learning
# Unsupervised Learning

# 1. Supervised Learning
# 2. Model Training and Model Evaluation
# 3. Continuous Target Variale
# One-Hot-Encoding:
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# Categorical variable y
y = [1, 2, 3, 4, 3, 2, 1]
# Reshape y to a 2D array (required by OneHotEncoder)
y_reshaped = np.array(y).reshape(-1, 1)
# Create an instance of OneHotEncoder
encoder = OneHotEncoder()
# Fit and transform the categorical variable
y_encoded = encoder.fit_transform(y_reshaped).toarray()
print(y_encoded)

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a Random-Forest object
rf = RandomForestRegressor()

# Model fit
rf.fit(X_train, y_train)

# Prediction
pred = rf.predict(X_test)

# Hyperparameter grid
hyperparameters = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Hyperparameter tuning, model fitting, and predicting
rf_cv = GridSearchCV(rf, hyperparameters, cv=10)
rf_cv.fit(X_train, y_train)
pred_rf = rf_cv.predict(X_test)

joblib.dump(rf_cv, 'rf_model.pkl')

# 4. Binary Target Variable
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
import joblib

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()

hyperparameters = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_cv = GridSearchCV(rf, hyperparameters, cv=10)
rf_cv.fit(X_train, y_train)

y_test_pred = rf_cv.predict(X_test)

y_prob = rf_cv.best_estimator_.predict_proba(X_test)

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test, y_test_pred)
print("Precision Score:")
print(precision)

joblib.dump(rf_cv, 'rf_classifier_model.pkl')

# 5. scikit.learn API
# estimator.fit(X, [y])
# ### Example estimators for supervised learning
# clf = RandomForestClassifier().fit(X_train, y_train) # Classification rgr = LinearRegression().fit(X_train, y_train) # Regression
# ### Example estimator for unsupervised learning
# kmeans = KMeans(n_clusters=5).fit(X_train) # Clustering
# ### Example estimators for data transformations
# pca = PCA().fit(X_train) # Dimensionality reduction
# scaler = preprocessing.StandardScaler().fit(X_train) # Preprocessing

# ### Example predictor for a classification task
# clf.predict(X_test)
# ### Example predictor for a regression task
# rgr.predict(X_test)
# ### Example predictor for a clustering task
# kmeans.predict(X_test)

# ### Example transformer for a dimensionality reduction task
# X_new = pca.transform(X_train)
# ### Example transformer for a preprocessing task
# X_scaled = scaler.transform(X_train) # Mean zero, variance one

# score = model.score(data)
# ### Example model for a classification task clf.score(X_test, y_test)
# ### Example model for a regression task
# rgr.score(X_test, y_test)

# 6. Advanced API
# Pipeline
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline = Pipeline([
    ("transformer", StandardScaler()),
    ("predictor", Ridge()),
])
pipeline.fit(X_train, y_train)
y_hat = pipeline.predict(X_test)

# Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
pipeline = Pipeline([
    ("transformer", ColumnTransformer([
        ("numerical", Pipeline([
            ("imputation", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), [0, 1, 2]),
        ("categorical", OneHotEncoder(), [3, 4, 5]),
    ])),
    ("predictor", Ridge()),
])

# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

# Cross-validated Grid Search
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

# Pipeline with Cross-validated Grid Search
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
# Preprocessing
scaler = StandardScaler()
# Dimensionality reduction
pca = PCA()
# Classification
svc = svm.SVC()
# Pipeline
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("svc", svc)])
# Parameters of pipelines can be set using '__' separated parameter names:
param_grid = {
         "pca__n_components": [5, 15, 30, 45, 60],
         "svc__C": 10.**np.arange(-3,3),
         "svc__gamma": 10.**np.arange(-3,3)
     }
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 4: Sklearn - continuous response variable
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pathlib
pathlib.Path('models').mkdir(parents=True, exist_ok=True)
pathlib.Path('plots').mkdir(parents=True, exist_ok=True)

#%% ------------------------------------------------------------------------------------
# BLOCK 1: Reading Dataset
# ------------------------
print('#'*50)
print('########## Reading Dataset ##########')
print('#'*50)

# Read the dataset from the URL via the following command
# 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
red = pd.read_csv(red_url, sep=';')

#%% ------------------------------------------------------------------------------------
# EX01: Set a global seed
print('---------- EX-01 ----------')
global_seed = 1418
np.random.seed(global_seed)

#%% ------------------------------------------------------------------------------------
# EX02: Split the dataset into
# - a pandas series 'y' with the target variable quality, and
# - a pandas data frame 'X' with all other variables
print('---------- EX-02 ----------')

X = red.drop('quality',  axis = 1)
y = red['quality']
print(X)
print(y)

#%% ------------------------------------------------------------------------------------
# EX03: Perform a train-test split with the ratio of 80:20
print('---------- EX-03 ----------')

# Hint: X_train, X_test, y_train, y_test = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#%% ------------------------------------------------------------------------------------
# BLOCK 2: Random Forest
# ----------------------
print('#'*50)
print('########## Random Forest ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Define a Random Forest Regressor with 100 trees
print('---------- EX-01 ----------')

rf = RandomForestRegressor(n_estimators = 100)

#%% ------------------------------------------------------------------------------------
# EX02: Fit the model on the training data
print('---------- EX-02 ----------')

rf.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX03: Make predictions on the test data and print the MSE
print('---------- EX-03 ----------')

pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred_rf)
print(mse_rf)

#%% ------------------------------------------------------------------------------------
# EX04: Define a dictionary that contains a set of possible values for the max_features and max_depth.
# Such a dictionary can be used for hypertuning via cross-validation
print('---------- EX-04 ----------')

hyperparameters_rf = {'max_features' :  ['auto', 'sqrt', 'log2'],
                      'max_depth':      [None, 5, 3, 1]}

#%% ------------------------------------------------------------------------------------
# EX05: Create a GridSearch CV object with 10-fold cross-validation and fit it on the training data
print('---------- EX-05 ----------')

cv_rf = GridSearchCV(rf, hyperparameters_rf, cv = 10, n_jobs=-1)
cv_rf.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX06: Print the best hyperparameters combination
print('---------- EX-06 ----------')


params_rf = cv_rf.best_params_
print(params_rf)


#%% ------------------------------------------------------------------------------------
# EX07: Let the best RandomForest model make predictions on the test data and print the MSE
print('---------- EX-07 ----------')

best_rf = cv_rf.best_estimator_
best_rf.fit(X_train, y_train)
pred_rf = cv_rf.predict(X_test)
pred_rf2 = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, pred_rf)
mse_rf2 = mean_squared_error(y_test, pred_rf2)
print(mse_rf)
print(mse_rf2)

#%% ------------------------------------------------------------------------------------
# BLOCK 3: Gradient Boosting
# --------------------------
print('#'*50)
print('########## Gradient Boosting ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Define a Gradient Boosting Regressor with 100 trees
print('---------- EX-01 ----------')

gb = GradientBoostingRegressor(n_estimators = 100)

#%% ------------------------------------------------------------------------------------
# EX02: Fit the model on the training data
print('---------- EX-02 ----------')

gb.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX03: Make predictions on the test data and print the MSE
print('---------- EX-03 ----------')

pred_gb = gb.predict(X_test)
mse_gb = mean_squared_error(y_test, pred_gb)
print(mse_gb)

#%% ------------------------------------------------------------------------------------
# EX04: Define a dictionary that contains a set of possible values for the max_features and max_depth.
# Such a dictionary can be used for hypertuning via cross-validation
print('---------- EX-04 ----------')

hyperparameters_gb = {'max_features' :  ['auto', 'sqrt', 'log2'],
                      'max_depth':      [None, 5, 3, 1]}

#%% ------------------------------------------------------------------------------------
# EX05: Create a GridSearch CV object with 10-fold cross-validation and fit it on the training data
print('---------- EX-05 ----------')

cv_gb = GridSearchCV(gb, hyperparameters_gb, cv = 10, n_jobs=-1)
cv_gb.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX06: Make predictions on the test data and print the MSE
print('---------- EX-06 ----------')

pred_gb = cv_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, pred_gb)
print(mse_gb)

#%% ------------------------------------------------------------------------------------
# BLOCK 4: Linear Model
# ---------------------
print('#'*50)
print('########## Linear Model ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Define a linear regression model with sklearn.linear_model.LinearRegression and fit it on the training data
print('---------- EX-01 ----------')

lm = LinearRegression()
lm = lm.fit(X_train, y_train)

#%% ------------------------------------------------------------------------------------
# EX02: Make predictions on the test data and print the MSE
print('---------- EX-02 ----------')

pred_lm = lm.predict(X_test)
mse_lm = mean_squared_error(y_test, pred_lm)
print(mse_lm)

#%% ------------------------------------------------------------------------------------
# BLOCK 5: Model Comparison
# -------------------------
print('#'*50)
print('########## Model Comparison ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Compare Random Forest, GBR, and Linear Model.
# Additionally, check the performance of a naive model (i.e., a model that always returns the mean of y.)
print('---------- EX-01 ----------')

print()
print('Random Forest:')
print('MSE: ' + str(mean_squared_error(y_test, pred_rf)))
print()
print('Gradient Boosting:')
print('MSE: ' + str(mean_squared_error(y_test, pred_gb)))
print()
print('Linear Model:')
print('MSE: ' + str(mean_squared_error(y_test, pred_lm)))
print()
print('Naive Model - Mean(y):')
print('MSE: ' + str(mean_squared_error(y_test, pd.Series(y_train.mean()).repeat(len(y_test)))))

#%% ------------------------------------------------------------------------------------
# EX02: Which model performs better in terms of the MSE?
print('---------- EX-02 ----------')

answer  = 'Random Forest is the best model here, close to gb.\n*Note:* Gradient Boosting might perform better if setting different seeds.'
print(answer)

#%% ------------------------------------------------------------------------------------
# EX03: Print the feature importance of the best model
print('---------- EX-03 ----------')

feat_imp = pd.DataFrame({'feature':     X_train.columns,
                         'importance':  cv_gb.best_estimator_.feature_importances_
                        })
print(feat_imp.sort_values(by = 'importance', ascending=False))

#%% ------------------------------------------------------------------------------------
# BLOCK 6: Model Evaluation
# -------------------------
print('#'*50)
print('########## Model Evaluation ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Adjust the following sample code to create a scatter plot of the predicted vs. the true values for the best model.
# You may need to firstly create a subfolder 'plots' in your working directory
print('---------- EX-01 ----------')

import matplotlib.pyplot as plt
f0 = plt.figure()
plt.scatter(y_test, pred_rf)
plt.title('Scatter plot prediction vs. truth')
plt.ylabel('Prediction')
plt.xlabel('y')
plt.show(block=False)
plt.pause(3)
plt.close()
f0.savefig('plots/scatter.png')


#%% ------------------------------------------------------------------------------------
# EX02: Customize the following sample code to sketch a histogram and a kernel density estimation of the prediction errors
print('---------- EX-02 ----------')

import seaborn as sns
f1 = plt.figure()
plt.title('Histogramm pred error best model')
sns.distplot(pred_rf - y_test, axlabel='Pred Error', kde=True, bins=25)
f1.savefig('plots/hist_kde.png')

f2 = plt.figure()
plt.title('Histogramm pred error mean model')
sns.distplot(y_train.mean() - y_test,
             axlabel='Pred Error',
             kde=False,
             bins=25)
f2.savefig('plots/hist_kde2.png')

#%% ------------------------------------------------------------------------------------
# BLOCK 7: Saving/Loading Model
# -----------------------------
print('#'*50)
print('########## Saving/Loading Model ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Save the trained model with joblib.dump() as a .pkl file for later use
print('---------- EX-01 ----------')

joblib.dump(cv_gb, 'models/regressor_gb.pkl')

#%% ------------------------------------------------------------------------------------
# EX02: How do you load the saved model?
print('---------- EX-02 ----------')

reg = joblib.load('models/regressor_gb.pkl')

