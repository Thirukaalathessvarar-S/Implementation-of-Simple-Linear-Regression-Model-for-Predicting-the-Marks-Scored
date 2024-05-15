# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import pandas, numpy and sklearn

2.Calculate the values for the training data set

3.Calculate the values for the test data set

4.Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thirukaalathessvarar S
RegisterNumber: 212222230161
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

##  splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

## graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### df.head()
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/390b0981-76a2-438b-85b6-3c6f3fc7376e)

### df.tail()
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/fb43eb76-3d5e-4be9-806a-e4c4769803e7)

### ARRAY VALUE OF X
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/c8aab348-27f3-4d11-b058-fe5ef0f99356)

### ARRAY VALUE OF Y
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/ffc573b5-d64b-4de1-b1c0-d96ac7088789)

### VALUES OF Y PREDICTION
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/7c76193a-ed42-4c61-aeab-ac188c1cc05d)

### ARRAY VALUES OF Y TEST
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/598227c6-0540-46f5-9a98-28fa9bc650b7)

### TRAINING SET GRAPH
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/982fb3dd-68fb-4835-85ea-beb8fcdf9d0d)

### TEST SET GRAPH
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/e692ad0a-25df-4608-a7c9-880bbbaaa9db)

### VALUES OF MSE,MAE AND RMSE
![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/98a62019-e8ff-4e80-9e1e-ec041d8e898b)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/4bcd9fd2-125c-4a87-8bf2-0a1580dcab2d)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708245/4ff3e450-ffda-480c-8f67-be608f8dc8b4)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
