import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
titanic_data = pd.read_csv('tested.csv')
print(titanic_data.head())
print(titanic_data.shape)
print(titanic_data.info())
print(titanic_data.isnull().sum())
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
print(titanic_data['Embarked'].mode()[0])

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
print(titanic_data.isnull().sum())
print(titanic_data.describe())
#finding the number of people survived and not survived
print(titanic_data['Survived'].value_counts())
#Data visualizing
sns.set()
print(sns.countplot(x='Survived',data=titanic_data))
plt.show()
print(titanic_data['Sex'].value_counts())
print(sns.countplot(x='Sex',data=titanic_data))
plt.show()
#number of survivors gender wise
print(sns.countplot(x='Sex', hue='Survived',data=titanic_data))
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Survivors by Gender')
plt.show()
print(sns.countplot(x='Pclass',data=titanic_data))
plt.show()
print(sns.countplot(x='Pclass', hue='Survived',data=titanic_data))
plt.show()
print(titanic_data['Embarked'].value_counts())
#converting categoricl columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
print(titanic_data.head())
X= titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape,X_train.shape,X_test.shape)
model = HistGradientBoostingClassifier()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

