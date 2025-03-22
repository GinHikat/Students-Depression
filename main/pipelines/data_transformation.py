import pandas as pd
import numpy as np
import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from main.extensions.utils.utils import save_object
from dataclasses import dataclass
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Define parameter for KNNImputer
KNN_params = {
    "missing_values": np.nan,
    'n_neighbors': 3,
    'weights': 'uniform'
}

@dataclass
class transformation_path:
    train_path = os.path.join('main/data/transformed', 'x_train.csv')
    test_path = os.path.join('main/data/transformed', 'x_test.csv')
    
    target_train_path = os.path.join('main/data/transformed', 'y_train.csv')
    target_test_path = os.path.join('main/data/transformed', 'y_test.csv')
    
    processor_path = os.path.join('main/artifacts', 'processor.pkl')
    
class transformation:
    def __init__(self):
        self.path = transformation_path()
        
    def transform(self, train:pd.DataFrame ,test: pd.DataFrame):
        try:
            cat_col = train.select_dtypes(include = 'object').columns
            num_col = train.select_dtypes(exclude = 'object').columns
            
            scaler = StandardScaler(with_mean=False)
            encoder = OneHotEncoder(handle_unknown='ignore')
            num_imputer = SimpleImputer(strategy='median')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            
            num_pipe = Pipeline(
                steps = [
                    ('imputer', num_imputer),
                    ('scaler', scaler)
                ]
            )
            
            cat_pipe = Pipeline(
                steps = [
                    ('imputer', cat_imputer),
                    ('encoder', encoder),
                    ('scaler', scaler)
                ]
            )
            
            #Get all into 1 column transformer object
            prep = ColumnTransformer(
                [
                    ('num_pipe', num_pipe, num_col),
                    ('cat_pipe', cat_pipe, cat_col)
                ]
            ) 
            
            logging.info("Filled Na, Encode and Scale complete!!")
            return prep
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def init_transform(self, target_col:str):
        try:
            logging.info("Load train and test raw") 
            
            #Read train and test from ingestion
            train = pd.read_csv('main/data/ingested/train.csv')
            test = pd.read_csv('main/data/ingested/test.csv')
            
            #Split target and variables
            
            target_train = train[[target_col]]
            train = train.drop([target_col], axis = 1)
            
            target_test = test[[target_col]]
            test = test.drop([target_col], axis = 1)
            
            
            logging.info('Separate and save target')
            
            #Call ColumnTransformer to transform data
            prep: ColumnTransformer = self.transform(train, test)
            
            logging.info("Start transforming")
            
            train = pd.DataFrame(prep.fit_transform(train))
            test = pd.DataFrame(prep.transform(test))
            
            #Save preprocessor object as pickle
            save_object(
                file_path = self.path.processor_path,
                obj = prep
            )

            logging.info('Save Preprocessor')
            
            #Save variables
            train.to_csv(self.path.train_path)
            test.to_csv(self.path.test_path)
            
            #Save target as independent files
            target_train.to_csv(self.path.target_train_path, header = True, index = False)
            target_test.to_csv(self.path.target_test_path, header = True, index = False)
        
            return train
        except Exception as e:
            raise CustomException(e, sys)
        
