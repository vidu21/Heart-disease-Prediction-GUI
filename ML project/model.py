import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import numpy as np
df= pd.read_csv('heart.csv')
df.drop_duplicates()
y=df['target']
X=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','slope','thal']]
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
X_scale=StandardScaler()
train_X=X_scale.fit_transform(X_train)
test_X=X_scale.fit_transform(X_test)
model=LogisticRegression()
model.fit(train_X,y_train)

pickle.dump(model, open('data.pkl','wb'))
data = pickle.load(open('data.pkl','rb'))

# Loading model to compare the results
data = pickle.load(open('data.pkl','rb'))

