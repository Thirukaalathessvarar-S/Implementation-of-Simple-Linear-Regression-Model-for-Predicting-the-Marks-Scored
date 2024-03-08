# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thirukaalathessvarar S
RegisterNumber:  212222230161
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
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

df.head()

![dfhead](https://user-images.githubusercontent.com/119393556/229976934-4d91a199-2a70-471e-ac2e-4f8652d84ea0.png)

df.tail()

![dftail](https://user-images.githubusercontent.com/119393556/229977042-8d6def67-5a4f-4662-8b58-9a90043ba4d8.png)

Array value of x

![xvalue](https://user-images.githubusercontent.com/119393556/229977127-908872b4-8375-4395-90f3-b9419455531f.png)

Array value of y

![yvalue](https://user-images.githubusercontent.com/119393556/229977197-3fb65de1-147b-48ab-b733-8d5f5fb95917.png)

Values of Y prediction

![ypred](https://user-images.githubusercontent.com/119393556/229977255-591059fe-5038-4752-a766-9bfe0ba8dfd6.png)

Array values of Y test

![ytest](https://user-images.githubusercontent.com/119393556/229977342-82be44e2-d510-4a85-8299-2b9ff562cfc9.png)

Training set graph

![train](https://user-images.githubusercontent.com/119393556/229977463-14f46551-c268-4c59-b3d3-3147b58864bb.png)

Test set graph

![test](https://user-images.githubusercontent.com/119393556/229977527-9e409d07-690a-4a61-86e9-3a17fa029aa4.png)

Values of MSE, MAE and RMSE

![mse](https://user-images.githubusercontent.com/119393556/229977591-53620b05-1dbf-40c5-a0dc-61dfa1c6a648.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
