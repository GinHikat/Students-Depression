from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from main.pipelines.full_process import Final
from main.pipelines.auto_predict import Data, Prediction

application = Flask(__name__)

app = application

final_executor = Final()
final_executor.all_step()

#Define routes
@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/prediction', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        data = Data(
            gender=request.form.get('Gender'),
            age=request.form.get('Age'),  
            academic_pressure=request.form.get('Academic Pressure'),
            study_satisfaction=request.form.get('Study Satisfaction'),
            sleep=request.form.get('Sleep Duration'),
            dietary=request.form.get('Dietary Habits'),
            suicidal_thought=request.form.get('Have you ever had suicidal thoughts ?'),  
            study_hour=request.form.get('Study Hours'),  
            financial_stress=request.form.get('Financial Stress'),  
            mental_illness=request.form.get('Family History of Mental Illness')
        )

        
        pred = data.get_data()
        predict = Prediction()
        result = predict.init_predict(pred)
        
        return render_template('result.html', result = result[0][1])
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000, debug=True)   
