# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 15:19:58 2018

@author: kishl
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset=pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting the linear regression model
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x,y)
exp = 6.5
exp_sal1= regressor1.predict(exp)

#Fitting the polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
lin_reg2.predict(poly_reg.fit_transform(6.5))
exp_poly=poly_reg.fit_transform(exp)
exp_sal2=lin_reg2.predict(exp_poly)


#Visualising the Linear Regressin results
plt.scatter(x,y,color='red')
plt.plot(x,regressor1.predict(x),color="blue")
plt.title('Truth or Bluff Linear Reg')
plt.xlabel("Experience")
plt.ylabel("Salary")

#Visualising the Polynmial Regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="blue")
plt.title('Truth or Bluff Polynomial Reg')
plt.xlabel("Experience")
plt.ylabel("Salary")

x_cont=np.arange(1,10,step=0.1)
x_cont=np.reshape(x_cont,(len(x_cont),1))
plt.plot(x_cont,lin_reg2.predict(poly_reg.fit_transform(x_cont)),color="green")