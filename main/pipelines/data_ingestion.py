import pandas as pd
import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from Source_info.source import Source

@dataclass #dataclass decorator replaces the init function for class object
class IngestionPath:
    train_path: str = Source.data_ingested_train_path
    test_path: str = Source.data_ingested_test_path
    
class DataIngestion:
    def __init__(self):
        self.path = IngestionPath()
    
    def init_ingestion(self):
        logging.info("Initialize data ingestion")

        try:
            df = pd.read_csv(Source.raw_data_path) 
            logging.info('Read data to csv')
            
            os.makedirs(os.path.dirname(self.path.train_path), exist_ok = True)
            ## Initialize folder to store dataset, even if it exists
            
            logging.info('Train test split')
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info("Saving train and test")
            train.to_csv(self.path.train_path, index = False, header = True)
            test.to_csv(self.path.test_path, index = False, header = True)
            
            logging.info("Saving complete!")
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    ingest = DataIngestion()
    ingest.init_ingestion()
