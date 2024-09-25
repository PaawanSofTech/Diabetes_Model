import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

df = pd.read_csv('diabetes.csv')

X = df[['Glucose', 'BMI', 'Age', 'BloodPressure']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

os.makedirs('models', exist_ok=True)

with open('models/diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
