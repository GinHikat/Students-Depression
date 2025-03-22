import pandas as pd
import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from dataclasses import dataclass
from main.extensions.utils.utils import load_object

@dataclass
class ModelPath:
    model_path = 'main/artifacts/model.pkl'
    preprocessor_path = 'main/artifacts/processor.pkl'
    
class Prediction:
    def __init__(self):
        self.path = ModelPath()
        
    def init_predict(self, data):
        try:
            model = load_object(self.path.model_path)
            processor = load_object(self.path.preprocessor_path)

            data_transformed = processor.transform(data)
            
            pred = model.predict_proba(data_transformed)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
          
class Data:
    def __init__(self, gender: str, age: int, academic_pressure: float, 
                 study_satisfaction: float, sleep: str, dietary: str, suicidal_thought: str,
                 study_hour: int, financial_stress: int, mental_illness: str):
        self.gender = gender
        self.age = age
        self.academic_pressure = academic_pressure
        self.study_satisfaction = study_satisfaction
        self.sleep = sleep
        self.dietary = dietary
        self.suicidal_thought = suicidal_thought
        self.study_hour = study_hour
        self.financial_stress = financial_stress
        self.mental_illness = mental_illness
        
    def get_data(self):
        try:
            data_input = {
                    'Gender': [self.gender],
                    'Age': [self.age],
                    'Academic Pressure': [self.academic_pressure],
                    'Study Satisfaction': [self.study_satisfaction],
                    'Sleep Duration': [self.sleep],
                    'Dietary Habits': [self.dietary],
                    'Have you ever had suicidal thoughts ?': [self.suicidal_thought],
                    'Study Hours': [self.study_hour],
                    'Financial Stress': [self.financial_stress],
                    'Family History of Mental Illness': [self.mental_illness]
                }
            data_input = pd.DataFrame(data_input)
            # data_input['Age'] = data_input['Age'].astype(int)
            # data_input['Academic Pressure'] = data_input['Academic Pressure'].astype(int)
            # data_input['Study Satisfaction'] = data_input['Study Satisfaction'].astype(int)
            # data_input['Study Hours'] = data_input['Study Hours'].astype(int)
            # data_input['Financial Stress'] = data_input['Financial Stress'].astype(int)
            return data_input
        except Exception as e:
            raise CustomException(e, sys)
            
    