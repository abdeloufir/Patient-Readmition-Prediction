from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## Route for a home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData