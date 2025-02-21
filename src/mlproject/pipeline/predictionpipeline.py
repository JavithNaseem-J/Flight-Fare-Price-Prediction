import os
import joblib
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        """
        Initialize the PredictionPipeline by loading the preprocessor and the trained model.
        """
        # Define paths for the preprocessor and model
        self.preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
        self.model_path = Path('artifacts/model_trainer/model.joblib')

        # Load the preprocessor and the trained model
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

    def preprocess_input(self, input_data):
        """
        Preprocess the input data using the preprocessor.
        Args:
            input_data (DataFrame): A single-row DataFrame containing feature values.
        Returns:
            ndarray: Preprocessed input data ready for prediction.
        """
        # Ensure the input data is in the correct format
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # Preprocess the input data
        processed_data = self.preprocessor.transform(input_data)

        return processed_data

    def predict(self, input_data):
        """
        Preprocess input data and make predictions.
        Args:
            input_data (DataFrame): A single-row DataFrame containing feature values.
        Returns:
            float: Predicted flight fare.
        """
        # Preprocess the input data
        processed_data = self.preprocess_input(input_data)

        # Make predictions using the trained model
        prediction = self.model.predict(processed_data)

        # Return the first (and only) prediction
        return prediction[0]