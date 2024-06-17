# Learning Goals
# Know the two main pandas data structures: Series and DataFrame
# Index Series and DataFrames properly
# Understand what data types can be stored in a Series or a DataFrame
# Understand the handling of missing values in pandas
# Know how to apply standard operations to Series and DataFrames
# Import and Export pandas DataFrames

# 1. Series
import numpy as np
import pandas as pd
### a) Create a series from a list:
# Create labels (list)
labels = ['a', 'b', 'c']
# Create data (list)
my_list = [10, 20, 30]
# Convert the list
pd.Series(data = my_list)
# Convert the list and the labels
pd.Series(data = my_list, index = labels)
# Shorter version; less clean but the same result:
print(pd.Series(my_list, labels))
### b) Create a series from a NumPy array:
# Convert a list to a NumPy array
arr = np.array(my_list)
# Convert a NumPy array to a Pandas series
ser = pd.Series(arr)
print(ser)
# Convert a NumPy array to a pandas series - with labels
ser = pd.Series(data = arr, index = labels)
print(ser)
### c) Create a series from a dictionary:
# Create a dictionary with similar data
d = {'a' : 10,
     'b' : 20,
     'c' : 30}
# Convert the dictionary to a series
ser = pd.Series(d)
print(ser)
### d) Data types in a series
# Besides numbers, strings can also be stored in a series...
ser_str = pd.Series(data = labels)
# ... also mixtures of numbers and strings:
print(pd.Series(data = [1, 'a']))
# ... even functions are possible
ser_func = pd.Series([sum,print])
print(ser_func)
# The data type of elements stored in the series can be seen
# either with printing the output or with .dtype
print(ser_str.dtype)
### e) Using index
# Create series 1
ser1 = pd.Series(data = [1,2,3,4],
                 index = ['FCB', 'B04', 'BVB', 'S04'])
# The index and the values can be returned separately:
print(ser1.index)
print(ser1.values)
# Access either with square brackets or index...
print(ser1['FCB']) # or ser1.loc['FCB']
# ... or a line number
print(ser1[0]) # or ser1.iloc[0]
# Create series 2
ser2 = pd.Series(data = [1,2,5,4],
                 index = ['FCB', 'B04', 'TSV', 'SVW'])
# Operations are performed based on the index:
summe = ser1 + ser2
print(summe)
# The missing value for pandas is np.nan, as in NumPy,
# even if it is displayed as nan in the print mode:
print(summe[1])
print(pd.isna(summe[1]))  # Warning=> not like this: summe[1] == np.nan
### f) Mixed data types in list, numpy array, pandas series
# Python list
liste = [1, 2, 'a']
# NumPy array
arr = np.array(liste)
# Pandas series
ser = pd.Series(liste)
print(liste[0])
print(arr[0] )
print(ser[0])
# Type conversion in python for String -> Numeric
print(int('1'))
# The pandas function .to_numeric converts the type of entire columns
print(pd.to_numeric(pd.Series(['1', '2', '3'])))
print(pd.to_numeric(pd.Series(['1', '2', '3'])).astype('float')) # int -> float
# Conversion Pandas Series -> Numpy Array
print(ser)
print(np.array(ser))
print(ser.to_numpy())
# Conversion Numpy Array -> List
print(list(np.array(ser)))
print(ser.to_list())

# 2. Data Frames
import numpy.random
rng = np.random.default_rng(123)
### a) Create a DataFrame:
df = pd.DataFrame(data = rng.standard_normal((3, 4)),
                  index = 'A B C'.split(),
                  columns = 'W X Y Z'.split())
print(df)
print(df.index)    # Row names
print(df.columns)  # Column names
print(df.values)   # Values only - without row and column names

### b) Selecting Columns
# Use single square brackets to select a column by its name
print(df['W'])
# Or several columns by their names: with a list
print(df[['W','Z']])
# Do not use...
print(df.W)
# ...because it is ambiguous and does not allow access to all column names
# Use .iloc to select columns based on their numbers;
# the colon indicates that all rows in that column will be selected
print(df.iloc[:,2])
#  Use .loc to select rows and columns based on their names
print(df.loc[:,'W'])
# Columns of a data frame are Series:
print(type(df['W']))

### c) Selecting rows
# Use .iloc to select rows based on their numbers
print(df.iloc[2])
# Use .loc to select rows based on their index (names)
print(df.loc['A'])
# or - cleaner
print(df.iloc[2, :])
print(df.loc['A', :])

### d) Selecting a subset of rows and columns
# With .loc you can select rows and columns,
# separated by commas based on the name
print(df.loc['B', 'Y'])
# ... also several from each
print(df.loc[['A','B'],
             ['W','Y']])
# ... including individual columns (3rd option for (b))
print(df.loc[:, 'W'])
# => With .loc and .iloc, a row is addressed only if one element is given
# With .iloc you can select rows and columns,
# separated by commas based on the indexes
print(df.iloc[1, 2])
# ... also several from each
print(df.iloc[[0, 1],
              [0, 2]])
# ... or:
print(df.iloc[0:2, 1:3])

### e) Conditional selection => like in R:
print(df)  # data frame from above
# Which values are positive?
print(df > 0)
# Keep only positive ones
print(df[df > 0])
# How many values are positive?
# ... per column
print((df > 0).sum())  # 'axis = 0' is the default
# ... per row
print((df > 0).sum(axis = 1))  # Sum in the column direction
# ... in total
print((df > 0).sum().sum())
# Keep all rows for which column W is positive
print(df[df['W'] > 0])
# Then, keep only column Y of that
print(df[df['W'] > 0]['Y'])
# Several conditions can be combined (as in R) with | and &
# Only rows for which column W is positive and column Y is positive
print(df[(df['W'] > 0) & (df['Y'] > 0)])

### f) Creating new columns
# With square brackets and a new column name
df['new'] = df['W'] + df['Y']
print(df)

### g) Deleting columns
df.drop(labels = 'new', axis = 1)
# Deletion is not "inplace" by default
print(df.columns)
# ... either save as new
df1 = df.drop(labels = 'new', axis = 1)
print(df1.columns)
# ... or set inplace = True
df.drop(labels = 'new', axis = 1, inplace = True)
print(df.columns)
# ... or with del: deletion is done "inplace"
del df['Z']
print(df.columns)

### h) Deleting rows
# Also with df.drop(), but axis = 0
print(df.drop(labels = 'C', axis = 0))
# => The inplace behavior is similar to the columns
print(df)

### i) Resetting the index
print(df)  # data frame from above
# With reset_index () the index is set back to 0, 1, ..., n-1
# and the original index is added as a new column
print(df.reset_index())
# drop = True prevents the old index from being added as a new column
df.reset_index(drop = True, inplace = True)
# Set new index: first the new index must be defined
newind = ['X', 'Y', 'Z']
print(df)  # Display the known data frame
# Add the desired index to the data frame as a column
df['newind'] = newind
# Then the new index can be made from this column with set_index
print(df.set_index('newind'))
# Alternative: pass NumPy array with the new index
print(df.set_index(np.array(newind)))

# 3. Missing Values
df = pd.DataFrame({'A' : [1, 2, np.nan],
                   'B' : [5, np.nan, np.nan],
                   'C' : [1, 2, 3]})
print(df)
### a) Removing missing values
print(df.dropna())  # default: axis = 0 => keep only complete rows
# For columns: axis = 1
print(df.dropna(axis = 1))

### b) Finding missing values
print(df.isna())

### c) Replacing missing values
print(df.fillna(value = 'ABC'))
# Replace NAs in column A with their mean value (not inplace)
print(df['A'].fillna(value = df['A'].mean()))

# 4. Data types
### a) Series have a single datatype
series = pd.Series([0, 1, 2])
print(series.dtype)
# If there are multiple data types in a series, it's dtype is objects
series = pd.Series([0, "1"])
print(series.dtype)

### b) A dataframe has one dtype per column
df = pd.DataFrame(
    {
        "strings": pd.Series(["a", "b", "c"], dtype="string"),
        "floats": pd.Series([0.1, 0.2, 0.3], dtype=float),
        "integers": pd.Series([0, 1, 2], dtype=int),
        "categorical": pd.Series(["hot", "cold"], dtype="category"),
        "date": pd.Series(pd.date_range("2023", freq="D", periods=3)),
        "bool": pd.Series([True, True, False], dtype=bool)
    }
)
print(df.dtypes)

### c) Integers, categories, etc. can contain NaN, too
df = pd.DataFrame(
    {
        "strings": pd.Series(["a", "b", "c", np.NaN], dtype="string"),
        "Int64": pd.Series([0, 1], dtype="Int64"),
        "int": pd.Series([0, 1], dtype=int),
        "categorical": pd.Series(["hot", "cold"], dtype="category"),
    }
)
print(df.dtypes)
print(df)

# Interlude - Missing Values Part 2
### Integers and strings cannot hold standard np.NaN
print(df["Int64"])

# 5. Groupby
### a) Grouping rows
# Create a data frame
data = {'Uni'    : ['LMU','LMU','TU','TU','KIT','KIT'],
        'Name'   : ['Ludwig','Omid','Tanja','Ursula','Kim','Ines'],
        'Groesse': [189,195,185,169,175,177]}
df = pd.DataFrame(data)
print(df)
# With groupby(), the data frame is grouped line by line,
# the output seems a little less meaningful
print(df.groupby('Uni'))

### b) Apply functions to groups
print(df.groupby('Uni')["Groesse"].mean())
by_uni = df.groupby('Uni')
print(by_uni["Groesse"].std())
print(by_uni.min())  # Warning: rows do not belong together!

# 6. DataFrame methods and attributes
df = pd.DataFrame({'col1' : [1,2,3,4],
                   'col2' : [444,555,666,444],
                   'col3' : ['abc','def','ghij','xyz']})
### a) Brief overview of the dataset
# Brief overview of data types, sizes, and missing values
print(df.info())
# Dimension of the dataset
print(df.shape)
# Display the first 5 rows
print(df.head())
# unique values
print(df['col2'].unique())
# Number of unique values
print(df['col2'].nunique())
# Frequency table
print(df['col2'].value_counts())
# Crosstab
print(pd.crosstab(df["col2"], df["col3"]))

### b) Application of functions
print(df['col1'] * 2)
# Length of the strings in column 3 per line
print(df['col3'].apply(len))
# Sum of column 1
print(df['col1'].sum())
# To have all the column sums, you don't need a loop:
print(df.sum())
# Warning: + is defined as concatenate for strings
print(df[["col1", "col2"]].sum(axis = 1))  # The string column has to be excluded

### c) sort by
# sort_values(): inplace = False is the default again
print(df.sort_values(by='col2'))

# 7. Input and Output
# Set the working directory at the top right (Spyder) or:
import os
os.chdir('/Users/maxmuster/python-kurs')
os.getcwd()
### a) csv
# Input
df = pd.read_csv('data/example.csv')
# Output
df.to_csv('data/example2.csv',
          index = False)

### b) Excel
# Input
pd.read_excel('data/Excel_Sample.xlsx',
              sheet_name = 'Sheet1')
# Output
df.to_excel('data/Excel_Sample2.xlsx',
            sheet_name = 'Sheet1')

# 8. Merging etc.
### append()
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
print(df)
df2 = pd.DataFrame([[5, 6]], columns=list('AB'))
print(df.append(df2))
#df.append(df2, ignore_index=True)


'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 3: Pandas
'''

import numpy as np
import pandas as pd

'''
The dataset that we will look at in this and the next few exercises contains information about different types of wine. 
A detailed description can be found at: 
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

In addition to some physical variables such as the alcohol content and 
pH value, the 'quality' column shows the wine quality as an average 
subjective assessment by at least three wine experts. In the next exercise, 
the quality of the wine is to be predicted based on the physical 
measured variables using various machine learning methods.
In this exercise, we get to know the dataset and practice the syntax and functionality of the pandas package.
'''
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

# Use the following command to output the number of rows and columns in the dataset
n_rows, n_cols = red.shape
print(n_rows, n_cols)

#%% ------------------------------------------------------------------------------------
# BLOCK 2: Selecting Specific Rows/Columns
# ----------------------------------------
print('#'*50)
print('########## Selecting Specific Rows/Columns ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Display the 'alcohol' column
print('---------- EX-01 ----------')

red['alcohol'] # option_1
red.loc[:,'alcohol'] # option_2

#%% ------------------------------------------------------------------------------------
# EX02: Display the first column
print('---------- EX-02 ----------')

red.iloc[:, 0]

#%% ------------------------------------------------------------------------------------
# EX03: Display the first 5 values of the last column
print('---------- EX-03 ----------')

red.iloc[:5, -1]

#%% ------------------------------------------------------------------------------------
# EX04: Display all values belonging to the row with index 1
print('---------- EX-04 ----------')

red.loc[1, :]

#%% ------------------------------------------------------------------------------------
# EX05: Display the first 5 values of the quality column
print('---------- EX-05 ----------')

red.loc[range(5), 'quality']

#%% ------------------------------------------------------------------------------------
# BLOCK 3: Conditional Selection of Rows
# --------------------------------------
print('#'*50)
print('########## Conditional Selection of Rows ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Display all values of all wines with an alcohol content of 14.9
print('---------- EX-01 ----------')

red[red['alcohol'] == 14.9].T

#%% ------------------------------------------------------------------------------------
# EX02: Display the alcohol content of all wines whose quality is 3
print('---------- EX-02 ----------')

red[red['quality'] == 3]['alcohol']

#%% ------------------------------------------------------------------------------------
# EX03: Display all values of all wines whose 'density' is greater than 0.999 and whose value for 'chlorides' is less than 0.065
print('---------- EX-03 ----------')

red[(red['density'] > 0.999) & (red['chlorides'] < 0.065)].T

#%% ------------------------------------------------------------------------------------
# BLOCK 4: Adding New Columns
# ---------------------------
print('#'*50)
print('########## Adding New Columns ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Create a new column in which the quality of the wine is binarized as follows:
#   quality >= 6    => 1
#   otherwise       => 0
print('---------- EX-01 ----------')

new_col = (red['quality']>=6)*1
red['quality_bin'] = new_col

#%% ------------------------------------------------------------------------------------
# BLOCK 5: Deleting Rows/Columns
# ------------------------------
print('#'*50)
print('########## Deleting Rows/Columns ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Delete the column 'free sulfur dioxide' and look at the result
print('---------- EX-01 ----------')

del red['free sulfur dioxide']
red.columns
red.shape

#%% ------------------------------------------------------------------------------------
# EX02: Delete the first 5 lines - but not inplace!
print('---------- EX-02 ----------')

red.drop(labels = range(5),
        axis = 0,
        inplace = False)

#%% ------------------------------------------------------------------------------------
# EX03: Delete all lines where quality == 3 - but not inplace!
print('---------- EX-03 ----------')

red[red['quality']!=3]

#%% ------------------------------------------------------------------------------------
# BLOCK 6: General Overview of Dataset
# ------------------------------------
print('#'*50)
print('########## General Overview of Dataset ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Provide a rough overview of the column names, the number of missing values, and the data types of each column
print('---------- EX-01 ----------')

red.info() #!EXONELINE

#%% ------------------------------------------------------------------------------------
# EX02: Display the first lines of the dataset to see all columns you have to adjust the pandas options:
# pd.set_option('display.max_columns', None)
print('---------- EX-02 ----------')

red.head() #!EXONELINE

#%% ------------------------------------------------------------------------------------
# EX03: Give a rough description of all columns
print('---------- EX-03 ----------')

red.describe() #!EXONELINE

#%% ------------------------------------------------------------------------------------
# EX04: Print the column names
print('---------- EX-04 ----------')

col_names = red.columns
print(col_names)

#%% ------------------------------------------------------------------------------------
# BLOCK 7: Overview of Target Variables
# -------------------------------------
print('#'*50)
print('########## Overview of Target Variables ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: How many different values does the variable 'quality' have?
print('---------- EX-01 ----------')

n_vals = red['quality'].nunique()
print(n_vals)

#%% ------------------------------------------------------------------------------------
# EX02: Print a frequency table for the variable 'quality', sorted by quality, not the frequency of occurrence
print('---------- EX-02 ----------')

freq_table = red['quality'].value_counts(sort=False).sort_index()
print(freq_table)

#%% ------------------------------------------------------------------------------------
# EX03: What is the data type of quality?
print('---------- EX-03 ----------')

answer = red['quality'].dtype
print(answer)

#%% ------------------------------------------------------------------------------------
# EX-additional: Search on the internet about making a bar plot; create a bar plot of the frequencies for the variable quality
print('---------- EX-additional ----------')

import matplotlib.pyplot as plt
plt.bar(red['quality'].value_counts(sort=False).index, red['quality'].value_counts(sort=False))
plt.show(block=False)
plt.pause(3)
plt.close()

#%% ------------------------------------------------------------------------------------
# BLOCK 8: Missing Values
# -----------------------
print('#'*50)
print('########## Missing Values ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Are there any missing values in the dataset?
print('---------- EX-01 ----------')

answer = 'No'
print(answer)
n_missings = red.isnull().sum()
print(n_missings)

#%% ------------------------------------------------------------------------------------
# BLOCK 9: Reset Index
# --------------------
print('#'*50)
print('########## Reset Index ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Display the index of the dataframe
print('---------- EX-01 ----------')

red.index

#%% ------------------------------------------------------------------------------------
# EX02: Set 'quality' as the new index - inplace
print('---------- EX-02 ----------')

red.set_index('quality', inplace = True)

#%% ------------------------------------------------------------------------------------
# EX03: Display the first few lines
print('---------- EX-03 ----------')

red.head()

#%% ------------------------------------------------------------------------------------
# EX04: Reset the index and keep the old index so that quality becomes a normal column again
print('---------- EX-04 ----------')

red.reset_index(drop = False, inplace = True)
red.head()


#%% ------------------------------------------------------------------------------------
# EX05: Did that work? Did you notice something?
print('---------- EX-05 ----------')

answer = 'Yes, only the order of the columns is different than before: Now, quality is the first column'
print(answer)

#%% ------------------------------------------------------------------------------------
# BLOCK 10: Converting Columns Units
# ----------------------------------
print('#'*50)
print('########## Converting Columns Units ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Convert 'citric acid' into whole number percentages and define the data type to be integer
print('---------- EX-01 ----------')

red['citric acid_100'] = (red['citric acid']*100).astype(int)
del red['citric acid']
red.head()
red.info()

#%% ------------------------------------------------------------------------------------
# BLOCK 11: Sum/Mean of Columns/Rows
# ----------------------------------
print('#'*50)
print('########## Sum/Mean of Columns/Rows ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Calculate the column sums and means
print('---------- EX-01 ----------')

res_sums = red.sum() #res_sums = red.sum(axis=1)
print(res_sums)
res_means = red.mean()
print(res_means)

#%% ------------------------------------------------------------------------------------
# BLOCK 12: Grouping
# ------------------
print('#'*50)
print('########## Grouping ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Calculate the average and maximum alcohol content for each quality level
print('---------- EX-01 ----------')

res_avg = red.groupby('quality')['alcohol'].mean()
print(res_avg)
res_max = red.groupby('quality')['alcohol'].max()
print(res_max)

#%% ------------------------------------------------------------------------------------
# EX-additional: Calculate the average quality level per 'citric acid_100'; present the results in a scatter plot
print('---------- EX-additional ----------')

mean_qu_acid = red.groupby('citric acid_100')['quality'].mean()
plt.close()
plt.scatter(mean_qu_acid.index, mean_qu_acid)
plt.show(block=False)
plt.pause(3)
plt.close()

#%% ------------------------------------------------------------------------------------
# BLOCK 13: Sorting
# -----------------
print('#'*50)
print('########## Sorting ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Sort the dataset in descending order of quality
print('---------- EX-01 ----------')

red.sort_values(by='quality', ascending=False)

#%% ------------------------------------------------------------------------------------
# BLOCK 14: Masking
# -----------------
print('#'*50)
print('########## Masking ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Replace all quality levels = 3 with the value 300
print('---------- EX-01 ----------')

red.loc[red['quality']==3,'quality'] = 300

#%% ------------------------------------------------------------------------------------
# EX02: Replace all quality levels = 300 with NAs (np.nan)
print('---------- EX-02 ----------')

red.loc[red['quality']==300,'quality'] = np.nan

#%% ------------------------------------------------------------------------------------
# EX03: How many missing values are in 'quality' now?
print('---------- EX-03 ----------')

n_missings = red['quality'].isnull().sum()
print(n_missings)

#%% ------------------------------------------------------------------------------------
# EX04: Fill in the missing values in 'quality' with the column median
print('---------- EX-04 ----------')

red['quality'].fillna(value = red['quality'].median(), inplace = True)

#%% ------------------------------------------------------------------------------------
# BLOCK 15: Saving Dataset
# ------------------------
print('#'*50)
print('########## Saving Dataset ##########')
print('#'*50)

#%% ------------------------------------------------------------------------------------
# EX01: Save the dataset as a csv file
print('---------- EX-01 ----------')

red.to_csv('../data/red.csv', index = True)

