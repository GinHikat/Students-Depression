import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging

from main.extensions.utils.utils import save_object, model_evaluation_reg, model_evaluation_class

@dataclass
class Train_path:
    model_path = os.path.join('main/artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.path = Train_path()
        
    def init_train(self):
        try:
            logging.info('Loading datasets')
            X_train = pd.read_csv('main/data/transformed/x_train.csv', index_col=0)
            X_test = pd.read_csv('main/data/transformed/x_test.csv', index_col=0)
            y_train = pd.read_csv('main/data/transformed/y_train.csv')
            y_test = pd.read_csv('main/data/transformed/y_test.csv')
            
            logging.info('Encoding target values')
            encoder = LabelEncoder()
            y_train = pd.DataFrame(encoder.fit_transform(y_train))
            y_test = pd.DataFrame(encoder.transform(y_test))
            
            model_best_params = {}
            model_result = {}
            
            logging.info('Define models and parameters')
            models = {
                "Logistic Regressor": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=128),
                "XGBClassifier": XGBClassifier(learning_rate = 0.1, n_estimators = 256), 
                "AdaBoost Classifier": AdaBoostClassifier(learning_rate=0.5, n_estimators=256),
                'CatBoosting Classifier': CatBoostClassifier(depth = 6, iterations = 100, learning_rate=0.1),
                'Gradient Boosting': GradientBoostingClassifier(learning_rate= 0.05, n_estimators = 256, subsample = 0.6)
            }
            
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regressor":{},
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            '''
            This part is for Hyperparameter Tuning
            
            for model_name, param in params.items():
                model = GridSearchCV(models[model_name], param, n_jobs=-1, verbose = False)
                model.fit(X_train, y_train)
                model_best_params[model_name] = model.best_params_
                
            return model_best_params
            '''
        
            '''
            This part is for Showing model performance
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, f1 = model_evaluation_class(y_test, y_pred)
                
                model_result[model_name] = accuracy
                model_resut = dict(sorted(model_result.items(), key=lambda item: item[1], reverse=True))
                return model_result  
            
            '''
            logging.info('Model training complete')
            
            model = AdaBoostClassifier(learning_rate=0.5, n_estimators=256)
            model.fit(X_train, y_train)
            
            logging.info('Saving model to artifacts')
            save_object(self.path.model_path, model)
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    train = ModelTrainer()
    x_train = train.init_train()
    
    