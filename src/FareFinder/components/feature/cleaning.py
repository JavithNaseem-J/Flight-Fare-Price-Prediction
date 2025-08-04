import os
import json
import numpy as np
import pandas as pd
from FareFinder.utils.logging import logger
from FareFinder.entities.config_entity import DataCleaningConfig


class Cleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Dropping columns: {self.config.columns_to_drop}")
        return df.drop(columns=self.config.columns_to_drop, errors='ignore')

    def convert_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        for col in self.config.datetime_columns:
            if col in df_copy.columns:
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col])
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        return df_copy

    def extract_time_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        time_mappings = {
            "Departure Date & Time": "Departure Time",
            "Arrival Date & Time": "Arrival Time"
        }

        for original_col, new_col in time_mappings.items():
            if original_col in df_copy.columns:
                if not pd.api.types.is_datetime64_dtype(df_copy[original_col]):
                    try:
                        df_copy[original_col] = pd.to_datetime(df_copy[original_col])
                    except Exception as e:
                        logger.warning(f"Could not convert {original_col} to datetime: {e}")
                        continue
                
                hour_col = f"{original_col}_hour"
                df_copy[hour_col] = df_copy[original_col].dt.hour

                conditions = [
                    (df_copy[hour_col] >= 6) & (df_copy[hour_col] < 12),
                    (df_copy[hour_col] >= 12) & (df_copy[hour_col] < 18),
                    (df_copy[hour_col] >= 18) & (df_copy[hour_col] < 24),
                    (df_copy[hour_col] >= 0) & (df_copy[hour_col] < 6)
                ]
                choices = ['Morning', 'Afternoon', 'Evening', 'Night']
                
                df_copy[new_col] = pd.Series(
                    np.select(conditions, choices, default='Unknown'), 
                    index=df_copy.index
                )
                
                df_copy.drop(columns=[hour_col], inplace=True)
                
        return df_copy

    def rename_target_column(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.config.target_column_mapping)
        
    def log_transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        target_column = "Total Fare"
        logger.info(f"Applying log transformation to target column: {target_column}")
        
        df_transformed = df.copy()
        
        if target_column not in df_transformed.columns:
            available_cols = df_transformed.columns.tolist()
            logger.error(f"Target column '{target_column}' not found. Available columns: {available_cols}")
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        df_transformed[target_column] = np.log1p(df_transformed[target_column])
        logger.info(f"Log transformation applied to {target_column}")
        
        return df_transformed

    def check_status(self):
        try:
            with open(self.config.file_status, 'r') as f:
                content = f.read().strip()
            
            if "Validation status:" in content:
                status_str = content.split("Validation status:")[1].strip()
                validation_status = status_str.lower() == 'true'
            else:
                try:
                    status_data = json.loads(content)
                    validation_status = status_data.get("Validation status", False)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse status file content: {content}")
                    validation_status = False
            
            logger.info(f"Data validation status: {validation_status}")
            return validation_status
        except Exception as e:
            logger.error(f"Error reading validation status: {e}")
            return False
    
    def clean_data(self):
        validation_status = self.check_status()
        
        if not validation_status:
            logger.error("Data validation failed. Skipping data cleaning.")
            raise ValueError("Data validation failed. Cannot proceed with data cleaning.")
        
        logger.info("Data validation passed. Proceeding with data cleaning.")
        logger.info(f"Reading data from {self.config.input_data_path}")

        df = pd.read_csv(self.config.input_data_path)
        
        if df is None or df.empty:
            logger.error("Input data is empty or None")
            raise ValueError("Input data is empty or None")
        
        logger.info(f"Original DataFrame shape: {df.shape}")
            
        df = self.convert_datetime_columns(df)
        df = self.extract_time_categories(df)            
        df = self.rename_target_column(df)            
        df = self.log_transform_target(df)            
        df = self.drop_columns(df)

        os.makedirs(os.path.dirname(self.config.cleaned_file), exist_ok=True)
        df.to_csv(self.config.cleaned_file, index=False)
        logger.info("Data cleaning completed successfully")
