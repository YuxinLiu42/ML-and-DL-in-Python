# Learning Goals
# Create and manipulate numpy arrays
# Understand numpy indexing, views and copies
# Conduct basic math operations on data

# 0. Import Packages
# In Python, packages are imported via 'import <package>',
# e.g., to import the package 'math':
import math
# Functions inside a package are addressed via
# package.function()
math.exp(1)
# ``````{python, echo=TRUE, eval=TRUE} invalid syntax
# You can assign a different name to a package via
# 'import <package> as <my_name>'
import math as m
# ... where you can choose the name with 'as'.
# You can then use this name to address the package. m.exp(1)
m.exp(1)
# ...it does not work without the package name (unlike R):
# exp(1)
## NameError: name 'exp' is not defined
# To load a certain function of a package: 'from <package> import <function>'
#  Advantages:
# - Fewer things are loaded; only those that are really needed
# - The function can be used directly with its name:
from math import exp
exp(1)

# 1. Install Packages
# Install a package in the console / command line / terminal via # 'pip install <package>'
# pip install opencv-python
# Help:
# pip install --help
# Access the command line directly from Python via '! <command>'
# !python --version

# 2. Package Math
# Constants
# math.pi
## NameError: name 'math' is not defined math.e
## NameError: name 'math' is not defined
# math.pow(2,3) # a to the power of b
## NameError: name 'math' is not defined
# math.sqrt(2) # square root
## NameError: name 'math' is not defined
# math.pow(8, 1/3) # arbitrary root
## NameError: name 'math' is not defined
# math.fabs(-3) # Absolute value
## NameError: name 'math' is not defined
# math.factorial(5) # Factorial
## NameError: name 'math' is not defined
# math.exp(1) # Exponential function
## NameError: name 'math' is not defined
# math.log(math.e) # Logarithm
## NameError: name 'math' is not defined
# math.log(8, 2) # Log base 2
## NameError: name 'math' is not defined
# math.cos(math.pi/3) # Trigonometric functions
## NameError: name 'math' is not defined

# 3. Numpy - Array Creation
import numpy as np
### a) Convert a list to a NumPy array => 'Vector'
my_list = [1, 2, 3]
print(np.array(my_list))
### b) Convert a list of lists to a NumPy array => 'Matrix'
my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(np.array(my_matrix))
### c) Arange: integers within an interval
# (the last element is excluded):
print(np.arange(0, 10))
# The distance between the integers is adjustable:
print(np.arange(0, 11, 2))
### d) Array with zeros
# 1D
print(np.zeros(3))
# 2D
print(np.zeros((5, 5)))
### e) Array with ones
# 1D
print(np.ones(3))
# 2D
print(np.ones((3, 3)))
### f) linspace(): equidistant numbers within an interval
# 3 numbers between 0 and 10 (10 is included)
print(np.linspace(0, 10, 3))
# To compare with arange():
# 3 is the distance, not 3 numbers
# Integers instead of floats
print(np.arange(0, 10, 3))
### g) eye(): identity matrix
print(np.eye(4))

# 4. Random Numbers
### a) rand(): an array with a specified size and
# values from a uniform distribution over [0,1)
# 1D
print(np.random.rand(2))
# 2D
print(np.random.rand(3, 3))
# 3D
print(np.random.rand(2, 2, 2))
### b) randn(): similar to rand() but the values from N(0,1)
print(np.random.randn(2))
### c) randint(): integers from range [a, b)
# 10 numbers between 1 and 100 (drawn with replacement)
print(np.random.randint(1, 100, 10))
### d) normal(): random numbers from N(mu, sigmaˆ2)-distrubtion
print(np.random.normal(2,2,5))
### e) poisson(): random numbers from a Poisson-distrubtion
print(np.random.poisson(5,10))
### f) shuffle(): random rearrangement of a sequence.
# It works 'in-place', i.e., the result need not be saved again
arr = np.arange(10)
print(arr)
np.random.shuffle(arr)
print(arr)
# Attention: the sorting is NOT done 'in-place'.
np.sort(arr)
print(arr)
### g) Use a random number generator for reproducibility
generator = np.random.default_rng(123)
print(generator.uniform())
print(generator.integers(low=0, high=10))
print(generator.standard_normal())
print(generator.choice(10, 10, replace=True))

# 5. Attributes & Methods
### a) reshape(): create an array with similar values,
# butinanewform
arr = np.arange(9)
print(arr)
print(arr.reshape(3, 3))
# The new dimension must match the original number of values
# (unlike R)
# print(arr.reshape(3, 4))
## ValueError: cannot reshape array of size 9 into shape (3,4)
### b) max(), min(), argmax(), argmin()
# 10 integers between 0 and 50
ranarr = np.random.randint(0,51,10)
print(ranarr)
print(ranarr.max())
print(ranarr.argmax())
print(ranarr.min())
print(ranarr.argmin())
### c) shape: returns the array dimension. An attribute, not a method.
# => therefore without brackets
print(arr.shape)
# Alternatively, the built-in function len() can be used
print(len(arr))
### d) The shape after reshape() :
# it is now a matrix with one row
# (recognizable by the double square brackets)
print(arr.reshape(1, 9))
print(arr.reshape(1, 9).shape)
# A matrix with one column:
print(arr.reshape(9, 1))
quadratic_arr = arr.reshape(3, 3)
print(quadratic_arr)
### e) max of 2-d matrix :
# is the max of all values in the matrix
quadratic_arr.max()
# We need to pass in the argument `axis` to get a column-wise reduction
print(quadratic_arr.max(axis=0))
# or row-wise reduction
print(quadratic_arr.max(axis=1))
### f) dtype: returns the data type of an array's values (attribute)
print(arr.dtype)
### f) type(): returns the type of a given input
# (not specific to NumPy)
print(type(arr))

# 6. Indexing and Slicing
# Numbers between 0 and 10
arr = np.arange(0,11)
print(arr)
### a) []-notation: very similar to Python lists
# selecting a single value
print(arr[8])
# selecting multiple values
print(arr[1:5])
### b) Broadcasting: in NumPy arrays, multiple elements can be assigned
# the same new value; this is not possible for Python lists:
# Set the first five values to 100...
arr[0:5] = 100
print(arr)
# ... does not work with lists
# plist[0:5] = 100
## NameError: name 'plist' is not defined
### c) Slicing: Python behaves a bit strange here:
# Reset the array to its original state:
arr = np.arange(0,11)
# Select the first 6 values:
slice_of_arr = arr[0:6]
# Change something in the slice, e.g., set everything to 99:
slice_of_arr[:] = 99
# Now something unexpected happens:
print(arr)
### d) copy(): a real copy operation ("Deep-Copy")
arr_copy = arr.copy()
arr[:] = -55
print(arr_copy)
### e) Indexing in 2D arrays, i.e. matrices
# Syntax: arr_ds[row_indices, col_indices]
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
print(arr_2d)
# Second row
print(arr_2d[1, :]) # Or, arr_2d[1]
# Element 2-1
print(arr_2d[1, 0])
# 2x2-slice from the top right
print(arr_2d[:2, 1:])
### f) 'Fancy Indexing': rows and columns can be selected
# in arbitrary order (similar to R).
# Create a new matrix and fill it with numbers
arr2d = np.zeros((5,5))
arr_length = arr2d.shape[1] # number of columns
# Fill the array
for i in range(arr_length):
         arr2d[i] = i
print(arr2d)

# You can now index arbitrarily:
print(arr2d[[3,4,2,1], :])
### g) Select elements by their properties rather than their indices
# Numbers between 1 and 7
arr = np.arange(1,8)
# Which elements are larger than 4?
print(arr > 4)
# For comparison: does not work with Python lists!
# plist = list(range(1,11))
# plist > 4
# Create a boolean vector
bool_arr = arr>4
# Select the elements that are greater than 4
print(arr[bool_arr])
# Shorter version:
print(arr[arr>4])

# 7. Arithmetic Operations
### a) Basic Arithmetic
arr = np.arange(0, 10)
arr + arr
arr * arr
arr - arr
arr + 3
arr**2
# Division 0/0 does not result in an error but a warning.
# It has been replaced by nan; more precisely by np.nan
div = arr/arr
## <string>:3: RuntimeWarning: invalid value encountered in divide
print(div)
# Division 1/0 does not result in an error but a warning.
# It has been replaced by infinity; more precisely by np.inf
infi = 1/arr
print(infi)
### b) Functions for arrays: standard functions can be applied directly
# to all (or some) elements of an array
# Square root
print(np.sqrt(arr))
# Exponential function
print(np.exp(arr))
# Maximum
print(np.max(arr))
# Sine
print(np.sin(arr))
# Logarithm
print(np.log(arr))
# or arr.max()
# For comparison: the math package methods don't work on Python lists
import math
# math.sqrt(plist)
### c) Use the @ operator for the dot product
a = np.arange(4).reshape(2, 2)
b = np.arange(4).reshape(2, 2)
# Simple multiplication
print(a * b)
# Dot product / matrix multiplication
print(a @ b)
### d) Transposing the matrix is also possible
a = np.arange(8).reshape(4, 2)
b = np.arange(8).reshape(4, 2)
# This won't work
# a@b
## ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
# You need to transpose the matrix first
print(a @ b.T)
print(a @ b.transpose())


"""
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 2: Numpy
"""

import numpy as np

#%% ------------------------------------------------------------------------------------
# BLOCK 1: ARRAY CREATION
# -----------------------
print('#'*50)
print('########## ARRAY CREATION ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Create a NumPy array with 10 zeros
print('---------- EX-01 ----------')

arr = np.zeros(10)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX02: Create a NumPy array with 10 ones
print('---------- EX-02 ----------')

arr = np.ones(10)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX03: Create a NumPy array with 10 fives
print('---------- EX-03 ----------')

arr = np.full(10, 5)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX04: Create a NumPy array with the integers from 10 to 50
print('---------- EX-04 ----------')

arr = np.arange(10, 51)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX05: Create a NumPy array with the even numbers from 10 to 50
print('---------- EX-05 ----------')

arr = np.arange(10, 51, 2)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX06: Create a 3x3 identity matrix
print('---------- EX-06 ----------')

arr = np.eye(3)

print(arr)

#%% ------------------------------------------------------------------------------------
# EX07: Create a NumPy array with 20 equidistant points
# between 0 and 1 (including the endpoints)
print('---------- EX-07 ----------')

arr = np.linspace(0, 1, 20)

print(arr)

#%% ------------------------------------------------------------------------------------
# BLOCK 2: RANDOM NUMBERS
# -----------------------
print('#'*50)
print('########## RANDOM NUMBERS ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Draw a random number from the uniform distribution [0, 1)
print('---------- EX-01 ----------')

generator = np.random.default_rng(123)
num = generator.uniform()

print(num)


#%% ------------------------------------------------------------------------------------
# EX02: Draw 25 random numbers from the N(0, 1) distribution
print('---------- EX-02 ----------')

num = generator.standard_normal(25)

print(num)

#%% ------------------------------------------------------------------------------------
# EX03: Draw 5 random integers between 2 and 10
print('---------- EX-03 ----------')

num = generator.integers(low=2, high=10, size=5)

print(num)


#%% ------------------------------------------------------------------------------------
# BLOCK 3: ATTRIBUTES & METHODS
# -----------------------------
print('#'*50)
print('########## ATTRIBUTES & METHODS ##########')
print('#'*50)


#%% ------------------------------------------------------------------------------------
# EX01: Randomly reorder the array rows. The row values should always stay together.
print('---------- EX-01 ----------')
# Given this 3x3 matrix:
arr = np.arange(9).reshape(3, 3)
generator.shuffle(arr)  # Random Shuffle

print(arr)

#%% ------------------------------------------------------------------------------------
# EX02: Create a 10x10 matrix with values ranging from 0.01 to 1 with step size 0.01
print('---------- EX-02 ----------')

arr = np.arange(0.01, 1.01, 0.01).reshape(10, 10)

print(arr)


#%% ------------------------------------------------------------------------------------
# BLOCK 4: INDEXING & SLICING
# ---------------------------
print('#'*50)
print('########## INDEXING & SLICING ##########')
print('#'*50)

# The following matrix is given:
mat = np.arange(1,26).reshape(5,5)

print('-- GIVEN MATRIX:')
print(mat)

#%% ------------------------------------------------------------------------------------
# EX01: Extract the 3x3 matrix in the lower right corner
print('---------- EX-01 ----------')

arr = mat[2:, 2:]

print(arr)

#%% ------------------------------------------------------------------------------------
# EX02: Access the value 20
print('---------- EX-02 ----------')

arr = mat[3, 4]

print(arr)

#%% ------------------------------------------------------------------------------------
# EX03: Extract the first 3 values of the 2nd column
print('---------- EX-03 ----------')

arr = mat[:3, 1:2]

print(arr)

#%% ------------------------------------------------------------------------------------
# EX04: Extract the 5th row
print('---------- EX-04 ----------')

arr = mat[4, :]

print(arr)


#%% ------------------------------------------------------------------------------------
# EX05: Extract all even numbers
print('---------- EX-05 ----------')

arr = mat[mat % 2 == 0]

print(arr)

#%% ------------------------------------------------------------------------------------
# BLOCK 5: ARITHMETIC OPERATIONS
# ------------------------------
print('#'*50)
print('########## ARITHMETIC OPERATIONS ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Calculate the sum of all the mat values
print('---------- EX-01 ----------')

res = np.sum(mat)

print(res)

#%% ------------------------------------------------------------------------------------
# EX02: Calculate the empirical standard deviation of the mat values
print('---------- EX-02 ----------')

res = np.std(mat)

print(res)

#%% ------------------------------------------------------------------------------------
# EX03: Calculate the sum of the mat columns.
print('---------- EX-03 ----------')

res = np.sum(mat, axis=0)

print(res)

#%% ------------------------------------------------------------------------------------
# EX04: Double all the mat values
print('---------- EX-04 ----------')

res = mat * 2

print(res)

#%% ------------------------------------------------------------------------------------
# BLOCK 6: Loading data files (advanced)
# --------------------------------------

#%% ------------------------------------------------------------------------------------
# EX01: Python's csv module
# Now that we have learned how to import modules, we can make use of Python's
# built-in csv reader. Below you can see two ways to use it. Use one of these
# to read the file
# 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# from last week into a numpy array
print('---------- EX-01 ----------')

import csv
with open("winequality-red.csv") as csv_file:
    reader = csv.DictReader(csv_file, delimiter=";")
    for row in reader:
        print(row)
        break

with open("winequality-red.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=";")
    for row in reader:
        print(row)
        break

dataset = []
with open("winequality-red.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=";")
    for i, row in enumerate(reader):
        if i == 0:
            continue
        else:
            dataset.append(row)
dataset = np.array(dataset, dtype=float)

print(dataset)

#%% ------------------------------------------------------------------------------------
# EX02: Loading the file with numpy's built-in loader
# Numpy provides multiple loaders. Check out np.loadtxt and configure its arguments
# to load the file at hand

dataset_via_np = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)

print(dataset_via_np)

# This next line check that the dataset read manually from the CSV and
# via numpy's loader are the same
np.testing.assert_array_almost_equal(dataset, dataset_via_np)

#%% ------------------------------------------------------------------------------------
# BLOCK 6: Some Data Science
# --------------------------

#%% ------------------------------------------------------------------------------------
# EX01: Calculate the average rating of the wines
quality = dataset_via_np[:, -1].mean()
assert quality == dataset_via_np.mean(axis=0)[-1]

print(quality)
np.testing.assert_almost_equal(quality, 5.63602251407)

#%% ------------------------------------------------------------------------------------
# EX02: Find all unique ratings
# Hint: check the numpy documentation for a function that provides this information
unique = np.unique(dataset_via_np[:, -1])
print(unique)

#%% ------------------------------------------------------------------------------------
# EX03: Extract all entries with the highest rating
subset = dataset_via_np[dataset_via_np[:, -1] == 8]
print(subset.shape)
assert subset.shape == (18, 12), subset.shape

#%% ------------------------------------------------------------------------------------
# EX04: Check if there are any NaNs in the dataset
print(np.any(np.isnan(dataset_via_np)))

#%% ------------------------------------------------------------------------------------
# BLOCK 7: Linear Regression
# --------------------------

#%% ------------------------------------------------------------------------------------
# EX01: Split the dataset into X and y
X = dataset_via_np[:, :-1]
y = dataset_via_np[:, -1]

#%% ------------------------------------------------------------------------------------
# EX02: Advanced Numpy: Least-Squares Regression
# Numpy provides many linear algebra routines in the subpackage np.linalg
# Find the appropriate function and obtain the coefficients `beta`
beta, _, _, _ = np.linalg.lstsq(X, y)

#%% ------------------------------------------------------------------------------------
# EX02: Predictions
# Compute the predictions as $X \beta$
y_hat = X @ beta

#%% ------------------------------------------------------------------------------------
# EX03: Compute the mean root mean squared error to judge the quality of the solution
rmse = np.sqrt(np.mean(np.power(y - y_hat, 2)))
print(rmse)

np.testing.assert_almost_equal(rmse, 0.6457934846531704)
