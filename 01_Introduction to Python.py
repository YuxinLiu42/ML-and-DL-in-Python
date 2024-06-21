# Learing Goals
# Know the basic data types in Python
# Know basics of the Python programming language
## indexing
## conditionals
## loops
## functions

# 1. Data Types
# a) Numbers
# Basic operations
# Addition
1 + 1
# Multiplication
1 * 3
# Division
1 / 2
# Modulo
4 % 2   # 0
5 % 2   # 1
# Bracket
(2 + 3) * (5 + 5)

# Assignment
x = 2
y = 3
z = x + y
z

# b) Strings
# Single Quotes
# 'single quotes'
# Double Quotes
# "double quotes"
# Mixed Form
# "It's easy to mix it up."
# Printing
x = 'Hello'
x

# Using variables in print function
num = 21
name = 'Emma'
print(f'My age is: {num}, my name is: {name}')
temperature = 21.4596
print(f'Today it is {temperature} degrees')
print(f'Today it is {temperature:.2f} degrees')

# The old ways...
num = 21
name = 'Emma'
print('My age is: {}, my name is: {}'.format(num,name))
print('My age is: {1}, my name is: {0}'.format(num,name))
print('My age is: {two}, my name is: {one}'.format(one=num,two=name))
print('My age is: ' + str(num) + ', my name is: ' + name)
print('My age is: %s, my name is: %s' % (num, name))

# "Adding strings"
first_name = 'Emma'
last_name = 'Watson'
name = first_name + last_name
print(f'My name is: {name}')
name = first_name + ' ' + last_name
print(f'My name is: {name}')
# You need to convert numbers to a string to add them
name = first_name + str(20)
# otherwise, this will fail
# name = first_name + 20

# c) Lists
# Lists are created with square brackets
# [1, 2, 3]
# Nesting is possible
# ['hi', 1, [1, 2]]
# For later reuse: let us assign a variable
my_list = ['a', 'b', 'c']
# Append another entry
my_list.append('d')
# Strings and numbers can be mixed
my_list.append(5)
# Show
print(my_list)

# Output the first element (attention: python index starts from 0 !!!)
print(my_list[0])
# Output the second element
print(my_list[1])
# Output the elements starting from the second one
print(my_list[1:])
# What is the result here (short exercise)?
print(my_list[:1])
print(my_list[-1])
print(my_list[-2:])
print(my_list[1::2])
print(my_list[1::-1])
print(my_list[4::-2])
my_list[0] = 'NEW'
print(my_list)

# Example of a nested list
nest = [1, 2, 3, [4, 5, ['target']]]
# The fourth element (with index 3) is a list
print(nest[3])
# The third element (with index 2) is a list
print(nest[3][2])
# The first element is string 'target'
print(nest[3][2][0])
# Strings behave similar to lists in python; you can access their elements via [
print(nest[3][2][0][2])
# What is the index of element '2'? (in R: which())
print(nest.index(2))

# d) Dictionaries
# A dictionary can be created via {}.
# Individual entries are separated by commas. # Each entry consists of a key-value pair.
d = {'key1': 'value1', 'key2': 'value2'}
print(d)
d2 = {'key1': 'value1', 'key2': 2}
print(d2)
# An element is accessed via the key
print(d['key1'])
print(d2['key2'])
# You can also use non-string keys
d = {1: 'value1', 2: 'value2'}
print(d)
d2 = {1: 1, 2: 2}
print(d2)
d3 = {-1: 0.5, 0: 0, 1: 0.5}
print(d3)
# Get all the dictionary keys:
print(d.keys())
# To access via indexes, the dict_keys should be first converted to a list:
print(list(d.keys())[1])
# Get all the dictionary values:
print(d.values())
# Get all the dictionary items, i.e. key-value pairs:
print(d.items())
# You can add elements to the dictionary
d['key3'] = 'value3'
# An important property to remember:
# Dictionaries are sorted by insertion order
d['key0'] = 'value0'
print(d)

# e) Tuples
# A tuple can be generated via ()
t = (1,2,3)
# A tuple is like a list, e.g., accessed via [] ...
print(t[0])
#... but its values cannot be overwritten
# t[0] = 'NEW'
# TypeError: 'tuple' object does not support item assignment

# f) Sets
# A set can be generated via {}
set1 = {1, 2, 3}
# A set ignores the repetitive elements
set2 = {1, 2, 3, 1, 2, 1, 2, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2}
print(set1)
print(set2)
# Function set() can be used to create a set from a list
set3 = set([1, 3, 5, 3, 1])
print(set3)

# Operations on sets:
# Union
print(set1 | set3)
# Intersection
print(set1 & set3)
# Difference
print(set1 - set3)
print(set3 - set1)
# Symmetric Difference
print(set1 ^ set3)

# g) Booleans
# True
# False

# 2. Comparison Operators
# Comparison operators are similar to the ones in R:
# Greater than
print(1 > 2)
# Less than
print(1 < 2)
# Greater or equal
print(1 >= 1)
# Less than or equal
print(1 <= 4)
# Equal
print(1 == 1)
# Unequal
print("hi" != "hey")

# 3. Logical Operators
# And
print((1 > 2) and (2 < 3))
# Or
print((1 > 2) or (2 < 3))
print((1 == 2) or (2 == 3) or (4 == 4))
# "Is an element of" 'x' in [1,2,3]
print('x' in ['x', 'y', 'z'])
# Use "not" for negation
print(not('x' in ['x', 'y', 'z']))
# And even more literal
print('x' not in ['x', 'y', 'z'])

# 4. if, else, elif statements
# Python uses very few brackets, because indentation is part of the syntax:
# If-Statement
if 1 < 2:
    print('Yes!')
# If - Else
if 1 < 2:
    print('first')
else:
    print('last')

if 1 == 2:
    print('first')
else:
    if 3 == 3:
        print('middle')
    else:
        print('last')

# Else if is abbreviated to elif
if 1 == 2:
    print('first')
elif 3 == 3:
    print('middle')
else:
    print('last')

# 5. Loops
# a) For-loop
# Define a list to iterate over:
seq = [1, 2, 3, 4, 5]
# The logic is as in R, but brackets are not needed; # indentation is part of the syntax:
for item in seq:
    print(item)

# b) while-loop
# The idea is as usual, again with sparse syntax
i = 1
while i < 5:
    print('i is now: {}'.format(i))
    i += 1  # or i = i+1
print('Loop is over, i is: ' + str(i))

# c) range()-Function
# range(n) creates an object that can be used for iteration;
# similar to seq_len(n) in R:
range(5)
for i in range(5):
    print(i)
# Unlike the seq_len(n), range(n) is not a vector / list.
# To print the elements, first convert range(n) into a list: r = range(5)
rl = list(range(5))
print(rl)
# Range from ... to ... (attention: the last element is excluded again)
print(list(range(5, 20)))

# d) enumerate()-Function
# enumerate(n) returns tuples that include the index of the current element
seq = ["cold", "lukewarm", "hot"]
for i, value in enumerate(seq):
    print(i, value)
# which is more readable than:
for i in range(len(seq)):
    print(i, seq[i])

for i in range(1, 101):
    if i % 9 == 0:
        print(f"{i} is divisible by 9")
    elif i % 17 == 0:
        print(f"{i} is divisible by 17")
    else:
        print(f"{i} is divisible by neither 9 nor 17")

# 6. List Comprehensions
# One cannot directly apply arithmetic operations to lists
# (unlike R-vectors):
x = [1, 2, 3, 4]
# print(x ** 2)
## TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
# A complicated alternative would be: (it can be even more complicated)
out = []
for item in x:
  out.append(item**2)
print(out)
# A shorter alternative is List Comprehensions;
# a shortened version of the for-loop:
y = [item ** 2 for item in x]
print(y)

# List comprehensions can be supplemented with "if" and "else":
y = [0 if item < 2.5 else 1 for item in x]
print(x)
print(y)
# And also to filter the items
y = [item for item in x if item % 2 == 0]
print(x)
print(y)

# 7. Functions
# A function is defined via the keyword 'def',
# followed by the function name and its parameters:
def my_func(param1 = 'default'):
    """
    Here the so-called docstring appears, a kind of help for functions.
    """
    print(param1)
# Show the function:
print(my_func)
# You can check the docstring of a function via help():
help(my_func)

# Execute the function with its default parameter:
my_func()
# Execute the function with a new parameter:
my_func('new param')
# When the function has more than one parameter, it is better to
#  explicitly state the parameter you specify
my_func(param1 = 'new param')
# A function with two parameters and return
def power(x, m = 2):
    return x**m
out = power(2, 3)
print(out)

# 8. Methods for Strings
# Everything in lower case
st = 'Hello world, Good Morning'
print(st.lower())
# Everything in capital letters
print(st.upper())
# Split words => list of individual words
print(st.split())
# Split on a certain character
print(st.split(','))
print(st.split(',')[1])



"""
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 1: Basics
"""

x = [1, 2, 3, 4, 5]

# %% ------------------------------------------------------------------------------------
# EX01: Extract the first 3 entries from list x
print('---------- EX-01 ----------')

sub_list = x[:3]

print(sub_list)

# %% ------------------------------------------------------------------------------------
# EX02: Extract the last 3 entries from list x
print('---------- EX-02 ----------')

sub_list = x[-3:]

print(sub_list)

# %% ------------------------------------------------------------------------------------
# EX03: Extract all entries from x except the last one
print('---------- EX-03 ----------')

sub_list = x[:-1]

print(sub_list)

# %% ------------------------------------------------------------------------------------
# EX04: Extract the word 'hello' from the list below
print('---------- EX-04 ----------')

lst = [1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7]

var = lst[3][1][2][0]

print(var)

# %% ------------------------------------------------------------------------------------
# EX05: Extract the word 'hard' from the following - badly formatted - dictionary
print('---------- EX-05 ----------')

d = {'oh': [1, 2, 3, {'man': ['why', 'is', 'this', {'so': [1, 2, 3, 'hard']}]}]}

var = d['oh'][3]['man'][3]['so'][3]

print(var)

# %% ------------------------------------------------------------------------------------
# EX06: Split the following string by space into a list of three elements
print('---------- EX-06 ----------')

s = 'Hello dear students!'

str_list = s.split(' ')

print(str_list)

# %% ------------------------------------------------------------------------------------
# EX07: Write a function `contains_ball` that returns True if its given input includes the word 'ball'
print('---------- EX-07 ----------')



def contains_ball(sentence: str) -> bool:
    return 'ball' in sentence

try:
    result = contains_ball('Is there a ball in the room here?')
    print(result)

    assert result, 'Your method does not return the correct result.'
except NameError:
    print('Seems like there is no "contains_ball" function.')


# %% ------------------------------------------------------------------------------------
# EX08: Write a function `count_ball` that counts the number of times
# the word 'ball' appears in a string.
print('---------- EX-08 ----------')




def count_ball(sentence: str) -> int:
    count = 0
    for word in sentence.lower().split():
        if word == 'ball':
            count += 1
    return count


def count_ball_smart(sentence: str) -> int:
    return sentence.count('ball')


try:
    result = count_ball('Is there a ball in the room here? Yes, a ball there is.')
    print(result)

    assert result == 2, 'Your method does not return the correct result.'
except NameError:
    print('Seems like there is no "count_ball" function.')

# %% ------------------------------------------------------------------------------------
# EX09: Write a function that reads a CSV file
print('---------- EX-09 ----------')

# For this we will use the wine quality dataset, which we will also use in later exercises
# Please download it from the UCI repository and put it in the same directory as the exercise
# 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

dataset = []

# The function open() opens a file in read mode. The file remains open in the following
# code block and is closed automatically when leaving the code block.
# The with open statement opens a file named winequality-red.csv and assigns the file object to open_file.
# The with statement ensures that the file is automatically closed after processing is complete.
with open('winequality-red.csv') as open_file:
    # You can iterate over the elements, which are the lines of the file
    for line in open_file:
        # The following line removes whitespace around the string and replaces all
        # double quotes by an empty string (i.e., removes the double quotes)
        line = line.strip().replace('"', '')
        # The strip() method is used to remove whitespace characters (including spaces, newlines, etc.)
        # at the beginning and end of a string.
        # replace('"', '') replaces all double quotes in a string with an empty string,
        # that is, removes all double quotes.
        # Your task is to convert the string into a list of individual elements
        # and cast these to the respective data type (float, except for quality, which is int)
        line = line.split(';')
        if 'quality' not in line:
            line = [float(element) for element in line]
            line[-1] = int(line[-1])
        dataset.append(line)

# %% ------------------------------------------------------------------------------------
# EX10: Finally, convert the dataset in a dictionary of dictionaries, so that we can access
# individual elements of the dataset as 'datasets[17]['quality'], which would return the
# quality of the 17th row (starting at 1)
print('---------- EX-10 ----------')
dataset_as_dict = {}
keys = dataset[0]
entry_number = 1

for entry in dataset[1:]:
    line_as_dictionary = {}
    for i in range(len(keys)):
        line_as_dictionary[keys[i]] = entry[i]
    dataset_as_dict[entry_number] = line_as_dictionary
    entry_number += 1

assert dataset_as_dict[17]['quality'] == 7

# Shorter solution
entry_number = 1
for entry in dataset[1:]:
    # Similar to a list comprehension, we can also do a dict comprehension
    line_as_dictionary = {keys[i]: entry[i] for i in range(len(keys))}
    dataset_as_dict[entry_number] = line_as_dictionary
    entry_number += 1

assert dataset_as_dict[17]['quality'] == 7

# And even shorter
for entry_number, entry in enumerate(dataset[1:], start=1):
    dataset_as_dict[entry_number] = {keys[i]: entry[i] for i in range(len(keys))}

assert dataset_as_dict[17]['quality'] == 7

# And finally, using the zip function
for entry_number, entry in enumerate(dataset[1:], start=1):
    dataset_as_dict[entry_number] = {key: entry for key, entry in zip(keys, entry)}

assert dataset_as_dict[17]['quality'] == 7

result = dataset_as_dict[17]['quality']
print(result)

assert result == 7, result
assert isinstance(result, int)
