from dataclasses import dataclass

@dataclass
class Source:
    raw_data_path: str = 'main/data/raw/data.csv' 
    data_ingested_test_path: str = 'main/data/ingested/test.csv'
    data_ingested_train_path: str = 'main/data/ingested/train.csv' 
    
    data_X_train_path: str = 'main/data/transformed/x_train.csv' 
    data_X_test_path: str = 'main/data/transformed/x_test.csv' 
    data_y_train_path: str = 'main/data/transformed/y_train.csv' 
    data_y_test_path: str = 'main/data/transformed/y_test.csv' 
    
    model_path: str = 'artifacts/model.pkl'
    processor_path: str = 'artifacts/processor.pkl'
    
    target_col: str = 'Depression'
