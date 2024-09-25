# diabetes_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Selecting the simplified feature set: Glucose, BMI, Age, BloodPressure
X = df[['Glucose', 'BMI', 'Age', 'BloodPressure']]
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model and scaler
with open('models/diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
with open('models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
