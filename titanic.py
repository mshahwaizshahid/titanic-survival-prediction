# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\projects\titanic\train.csv")

#Exploaing dataset
print(df.head())

# Checking missing values
print(df.isnull().sum())

# Filling missing Age values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Filling missing Embarked values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Dropping unnecessary columns
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Converting categorical data into numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Define Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Building Classification Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

# User Input
print("\n===== Titanic Survival Prediction =====")

pclass = int(input("Enter Passenger Class (1,2,3): "))

sex = input("Enter Gender (male/female): ")

if sex == "male":
    sex = 0
else:
    sex = 1

age = float(input("Enter Age: "))

sibsp = int(input("Enter Number of Siblings/Spouse: "))

parch = int(input("Enter Number of Parents/Children: "))

fare = float(input("Enter Fare (0-512): "))

embarked = input("Enter Embarked (S/C/Q): ")

if embarked == "S":
    embarked = 0
elif embarked == "C":
    embarked = 1
else:
    embarked = 2

# Prediction Data
user_data = pd.DataFrame([[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked
]], columns=X.columns)
# Predict using user input
prediction = model.predict(user_data)

# Result
print("\nPrediction Result:")

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")

# Evaluation
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)