import pandas as pd
import numpy as np
import json
import joblib   
from pathlib import Path
from FareFinder.utils.logging import logger
from FareFinder.entities.config_entity import ModelEvaluationConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        try:
            if not Path(self.config.test_raw_data).exists():
                raise FileNotFoundError(f"Test data file not found at {self.config.test_raw_data}")
            if not Path(self.config.preprocessor_path).exists():
                raise FileNotFoundError(f"Preprocessor file not found at {self.config.preprocessor_path}")
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Model file not found at {self.config.model_path}")

            logger.info("Loading preprocessor and model...")
            preprocessor = joblib.load(self.config.preprocessor_path)
            model = joblib.load(self.config.model_path)

            if hasattr(model, 'best_estimator_'):
                logger.info("Model is a RandomizedSearchCV object, extracting best estimator...")
                best_params = model.best_params_
                model = model.best_estimator_
            else:
                best_params = model.get_params()
                logger.info("Model is a direct estimator, using its parameters...")


            test_data = pd.read_csv(self.config.test_raw_data)
            target_column = self.config.target_column

            if target_column not in test_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in test data.")

            test_x = test_data.drop(columns=[target_column])
            test_y = test_data[target_column]

            test_x_preprocessed = preprocessor.transform(test_x)

            logger.info("Making predictions on the test data...")
            predictions = model.predict(test_x_preprocessed)

            logger.info("Evaluating model performance...")
            mse = mean_squared_error(test_y, predictions)
            mae = mean_absolute_error(test_y, predictions)
            r2 = r2_score(test_y, predictions)
                

            metrics = {
                    "mean_squared_error": mse,
                    "mean_absolute_error": mae,
                    "r2_score": r2
            }
                
            logger.info(metrics)


            metrics_file = Path(self.config.root_dir) / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
            logger.info(f"Metrics saved to {metrics_file}")


            return metrics
        
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e