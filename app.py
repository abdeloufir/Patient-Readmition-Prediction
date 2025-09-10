from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

## Route for a home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            mb_readmitted_gt30_ct = int(request.form.get('mb_readmitted_gt30_ct')),
            mb_readmitted_lt30_ct = int(request.form.get('mb_readmitted_lt30_ct')),
            mb_readmitted_no_ct = int(request.form.get('mb_readmitted_no_ct')),
            distinct_diag_count = int(request.form.get('distinct_diag_count')),
            encounter_ct = int(request.form.get('encounter_ct')),
            number_inpatient = int(request.form.get('number_inpatient')),
            mb_number_inpatient_ct = int(request.form.get('mb_number_inpatient_ct')),
            mb_number_diagnoses_ct = int(request.form.get('mb_number_diagnoses_ct')),
            admission_type = request.form.get('admission_type'),
            A1Cresult = request.form.get('A1Cresult')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)