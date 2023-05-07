import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


df = pd.read_csv('training.csv')
df_idRemoved = df.drop('id', axis=1)
data = df_idRemoved[['having_IP_Address', 'URL_Length', 'Domain_registeration_length','age_of_domain', 'web_traffic', 'Result']].copy()
x = data.iloc[:, :-1]
y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
ypred = model.predict([[1, 1, -1, 1, -1]])
if ypred[0] == -1:
  print("this website is safe")
else:
  print("this website is not safe")

filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)