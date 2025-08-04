import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from FareFinder.utils.helpers import read_yaml
from FareFinder.constants import *

class PredictionPipeline:
    def __init__(self):
        self.schema = read_yaml(Path('schema.yaml'))

        self.cat_cols = self.schema.categorical_columns
        self.num_cols = self.schema.numeric_columns
        self.target_column = self.schema.target_column.name

        self.preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
        self.model_path = Path('artifacts/model_trainer/model.joblib')
        self.label_encoders_path = Path('artifacts/data_transformation/label_encoders.pkl')

        for path in [self.preprocessor_path, self.model_path, self.label_encoders_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path)

    def preprocess_input(self, input_data):
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        input_data.columns = input_data.columns

        required_columns = self.preprocessor.feature_names_in_
        missing_columns = [col for col in required_columns if col not in input_data.columns]

        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")

        data = input_data.copy()

        for column in self.num_cols:
            if column in data.columns:
                data[column] = data[column].astype(float)

        # Encode categorical features using saved label encoders
        for column in self.cat_cols:
            if column in data.columns and column in self.label_encoders:
                encoder = self.label_encoders[column]
                unknown_labels = set(data[column].astype(str)) - set(encoder.classes_)
                if unknown_labels:
                    raise ValueError(f"Unknown categories in column '{column}': {unknown_labels}")
                data[column] = encoder.transform(data[column].astype(str))

        try:
            processed_data = self.preprocessor.transform(data)
            return processed_data
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {str(e)}")

    def predict(self, input_data):
        try:
            processed_data = self.preprocess_input(input_data)
            prediction = self.model.predict(processed_data)
            prediction = np.expm1(prediction)
            return prediction[0]
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
