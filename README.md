# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: Start the program

step 2: import the needed packages

step 3: Assigning hours to x and scores to y.

step 4:plot the scatter plot

step 5: Used mse,rmse,mae formula to find the values.

step 6: End the program
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Jayasuryaa k

RegisterNumber: 212222040060
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
  

```


## Output:
## plot
![Screenshot 2024-08-23 133814](https://github.com/user-attachments/assets/15d085b3-338c-4922-87c2-99c39208110b)  



![Screenshot 2024-08-23 133927](https://github.com/user-attachments/assets/d312e28f-7c58-401e-a99f-e57f75cc0424)


## Values of MSE,MAE, and RMSE:
![Screenshot 2024-08-28 110741](https://github.com/user-attachments/assets/15c7e860-f186-49ed-9b0a-fde9b8ad3bad)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
