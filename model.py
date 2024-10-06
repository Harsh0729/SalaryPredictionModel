# Importing necessary libraries
import numpy as np  # For mathematical operations
import matplotlib.pyplot as plt  # For plotting graphs (not used here, but commonly imported for data visualization)
import pandas as pd  # For data manipulation (handling data in table form)
import pickle  # For saving and loading the trained model

# Step 1: Loading the dataset
# The dataset 'hiring.csv' contains hiring-related data (experience, test scores, and salary).
dataset = pd.read_csv('hiring.csv')

# Step 2: Handling missing data
# Filling missing values in the 'experience' column with 0, since missing experience likely indicates no experience.
dataset['experience'].fillna(0, inplace=True)

# Filling missing values in the 'test_score' column with the mean test score.
# This is a common approach to handle missing numerical data.
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# Step 3: Extracting the features (X) and target variable (y)
# Features (X): We are selecting the first three columns (experience, test_score, interview_score) for training.
# 'iloc[:, :3]' selects all rows and the first 3 columns.
X = dataset.iloc[:, :3]

# Step 4: Converting categorical data (experience in words) to numerical data
# Defining a function to convert experience written in words (e.g., 'two') to numbers (e.g., 2).
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}  # Mapping words to numbers
    return word_dict[word]  # Returning the corresponding number for the word

# Applying the conversion function to the 'experience' column
X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

# Target variable (y): This is the last column in the dataset, which represents the salary.
y = dataset.iloc[:, -1]

# Step 5: Splitting the dataset into training and testing data
# In this case, the entire dataset is small, so we're using all of it for training.
# (Normally, you'd split the data into training and testing sets, but not here.)

# Step 6: Training the model using Linear Regression
# Importing the LinearRegression class from sklearn (a popular machine learning library)
from sklearn.linear_model import LinearRegression

# Creating an instance of the LinearRegression model
regressor = LinearRegression()

# Training the model on the data (fitting the regression model with the features X and target variable y)
regressor.fit(X, y)

# Step 7: Saving the trained model to a file
# Using 'pickle' to serialize and save the model to disk for future use.
pickle.dump(regressor, open('model.pkl', 'wb'))  # Saving the model as 'model.pkl'

# Step 8: Loading the saved model and making predictions
# Loading the model back from the saved file to make predictions.
model = pickle.load(open('model.pkl', 'rb'))

# Making a prediction for a new employee with 2 years of experience, 9 test score, and 6 interview score.
# The 'predict' function outputs the predicted salary based on the input values.
print(model.predict([[2, 9, 6]]))
