# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from model import predict_price

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        data = request.form.to_dict()
        data['Year'] = int(data['Year'])
        new_data = {
            'Property Type': [data['Property Type']],
            'Area': [data['Area']],
            'Region': [data['Region']],
            'Year': [data['Year']],
            'Home Type': [data['Home Type']]
        }

        prediction = predict_price(new_data)  # Assuming predict_price function is defined
        return render_template('predict.html', prediction_text=f'Predicted Price: {prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
