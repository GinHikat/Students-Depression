import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV

from main.extensions.exceptions.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) #file_path will be defined to artifacts folder
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            
            dill.dump(obj, file) #Dill convert the object into byte string in file
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file) #Dill converts back from byte string 
                                    #into object to be used
    except Exception as e:
        raise CustomException(e, sys)
    
def model_evaluation_class(y_test, y_pred): 
    try:
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return f1, accuracy
    except Exception as e:
        raise CustomException(e, sys)
     
def model_evaluation_reg(y_test, y_pred): 
    try:
        r2 = r2_score(y_test, y_pred)
        rmse= root_mean_squared_error(y_test, y_pred)
        return r2, rmse
    except Exception as e:
        raise CustomException(e, sys)