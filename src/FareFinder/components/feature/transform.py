import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from FareFinder.utils.logging import logger
from sklearn.pipeline import Pipeline
from FareFinder.entities.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split




class Transformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.target_column = config.target_column
        self.label_encoders = {}
        
        self.categorical_columns = config.categorical_columns
        self.numerical_columns = config.numerical_columns
        

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.copy()
            
                        
            if self.target_column not in data.columns:
                if 'Total Fare (BDT)' in data.columns:
                    logger.info(f"Renaming 'Total Fare (BDT)' to '{self.target_column}'")
                    data.rename(columns={'Total Fare (BDT)': self.target_column}, inplace=True)
                else:
                    raise ValueError(f"Target column '{self.target_column}' not found in data")
            
            for column in self.categorical_columns:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le
            
            os.makedirs(os.path.dirname(self.config.label_encoder), exist_ok=True)
            joblib.dump(self.label_encoders, self.config.label_encoder)
            logger.info(f"Saved label encoders to {self.config.label_encoder}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            raise e

    def train_test_splitting(self):
        try:
            logger.info(f"Loading data from {self.config.data_path}")
            data = pd.read_csv(self.config.data_path)
            
            data = self.preprocess_data(data)
            
            train, test = train_test_split(data, test_size=0.25, random_state=42)

            os.makedirs(self.config.root_dir, exist_ok=True)
            
            train_path = Path(self.config.root_dir) / "train.csv"
            test_path = Path(self.config.root_dir) / "test.csv"
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Split data into training and test sets")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")

            return train, test
            
        except Exception as e:
            logger.error(f"Error in train_test_splitting: {e}")
            raise e
    
    def preprocess_features(self, train, test):
        try:
            # Identify numerical and categorical columns
            numerical_columns = train.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = train.select_dtypes(include=["object", "category"]).columns

            if self.target_column in numerical_columns:
                numerical_columns = numerical_columns.drop(self.target_column)

            logger.info(f"Numerical columns: {list(numerical_columns)}")
            logger.info(f"Categorical columns: {list(categorical_columns)}")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns)
                ],
                remainder="passthrough"
            )

            # Separate features and target
            train_x = train.drop(columns=[self.target_column])
            test_x = test.drop(columns=[self.target_column])
            train_y = train[self.target_column]
            test_y = test[self.target_column]

            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            # Ensure target is 2D array
            train_y = train_y.values.reshape(-1, 1)
            test_y = test_y.values.reshape(-1, 1)

            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")

            np.save(Path(self.config.root_dir) / "train_processed.npy", train_combined)
            np.save(Path(self.config.root_dir) / "test_processed.npy", test_combined)

            logger.info("Preprocessed train and test data saved successfully.")
            return train_processed, test_processed

        except Exception as e:
            logger.error(f"Error in preprocess_features: {e}")