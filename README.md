# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: K SHAKTHI SUNDAR
RegisterNumber:  212222040152
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt  # Import plt for plotting
from sklearn import metrics

# Read the CSV file
data = pd.read_csv("/content/Salary.csv")

# Display the first few rows of the data
print(data.head())

# Get information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Encode the 'Position' column
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

# Select features (X) and target variable (y)
x = data[["Position", "Level"]]
y = data["Salary"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize the Decision Tree regressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# Calculate mean squared error
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared score
r2 = metrics.r2_score(y_test, y_pred)
print("R-squared Score:", r2)

# Make a prediction on new data
prediction = dt.predict([[5, 6]])
print("Predicted Salary:", prediction)

# Plot the decision tree
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/0035654d-678e-4cfd-9cea-8e4a7fbedca1)

![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/14a1b5ba-2cc3-4511-9481-0978ac2ae39b)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/cc77766c-2538-45ee-b352-b1400850e987)

![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/67116f0e-559e-4f48-a381-1ef1831ad86c)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/46f393ca-3a93-4bf4-9d9c-96de10819327)
![image](https://github.com/UdhayanithiM/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/127933352/9b0445fb-b9dd-48fb-b46e-baa8695e3676)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
