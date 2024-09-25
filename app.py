from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    glucose = float(request.form['glucose'])
    bloodpressure = float(request.form['bloodpressure'])
    bmi = float(request.form['bmi'])
    age = float(request.form['age'])

    input_data = np.array([[glucose, bloodpressure, bmi, age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    result = "You might have diabetes." if prediction[0] == 1 else "You are less likely to have diabetes."

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
