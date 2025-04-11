import sys, os
from main.extensions.exceptions.exception import CustomException
from main.extensions.logging.logger import logging
from main.pipelines.data_ingestion import DataIngestion
from main.pipelines.data_transformation import transformation
from main.pipelines.model_training import ModelTrainer
from dataclasses import dataclass
from datetime import datetime as dt
from main.cloud.s3_syncer import *
from Source_info.source import Source

# @dataclass
# class Config:
#     s3_sync = S3Sync()
    
#     def gettime(self):
#         timestamp = dt.now()
#         timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
#         return timestamp
    
#     # local artifact is going to s3 bucket    
#     def sync_artifact_dir_to_s3(self):
#         try:
#             self.bucket_name = "StudentDepression"
#             self.artifact_dir = 'main/artifacts'
            
#             aws_bucket_url = f"s3://{self.bucket_name}/artifact/{self.gettime()}"
#             self.s3_sync.sync_folder_to_s3(folder = self.artifact_dir, aws_bucket_url=aws_bucket_url)
#         except Exception as e:
#             raise CustomException(e,sys)
        
#     ## local final model is going to s3 bucket 
        
#     def sync_saved_model_dir_to_s3(self):
#         try:
#             aws_bucket_url = f"s3://{self.bucket_name}/final_model/{self.gettime()}"
#             self.s3_sync.sync_folder_to_s3(folder = self.artifact_dir,aws_bucket_url=aws_bucket_url)
#         except Exception as e:
#             raise CustomException(e,sys)
        


class Final:
    def __init__(self):
        pass
    
    def all_step(self):
        # self.ingest = DataIngestion()
        # self.ingest.init_ingestion()
        
        # self.transform = transformation()
        # self.transform.init_transform(Source.target_col)
        
        self.trainer = ModelTrainer()
        self.trainer.init_train()
        
        # self.sync = Config()
        # self.sync.sync_artifact_dir_to_s3()
        # self.sync.sync_saved_model_dir_to_s3()
        
        
