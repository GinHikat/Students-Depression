import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from main.pipelines.data_ingestion import DataIngestion
from main.pipelines.data_transformation import transformation
from main.pipelines.model_training import ModelTrainer

class Final:
    def __init__(self):
        pass
    
    def all_step(self):
        self.ingest = DataIngestion()
        self.ingest.init_ingestion()
        
        self.transform = transformation()
        self.transform.init_transform('Depression')
        
        self.trainer = ModelTrainer()
        self.trainer.init_train()
        
        
