import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('iris.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df['variety'].value_counts())
print(df.isnull().sum())
print(df['sepal.length'].hist())
print(df['petal.length'].hist())
print(df['petal.width'].hist())
print(df['sepal.width'].hist())
#scatterplot
colors=['red','orange','green']
variety=['Setosa','Versicolor','Virginica']
for i in range(3):
    x=df[df['variety']==variety[i]]
    plt.scatter(x['sepal.length'],x['sepal.width'],c=colors[i],label=variety[i])
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
for i in range(3):
    x=df[df['variety']==variety[i]]
    plt.scatter(x['petal.length'],x['petal.width'],c=colors[i],label=variety[i])
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()
for i in range(3):
    x=df[df['variety']==variety[i]]
    plt.scatter(x['sepal.length'],x['petal.length'],c=colors[i],label=variety[i])
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend()
for i in range(3):
    x=df[df['variety']==variety[i]]
    plt.scatter(x['sepal.width'],x['petal.width'],c=colors[i],label=variety[i])
    plt.xlabel('sepal width')
    plt.ylabel('petal width')
    plt.legend()
from sklearn.model_selection import train_test_split
X=df.drop(columns=['variety'])
Y=df['variety']
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.30)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
print("Accuracy:",model.score(x_test,y_test)*100)







