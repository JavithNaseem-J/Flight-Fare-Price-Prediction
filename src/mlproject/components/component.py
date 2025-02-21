import os
import urllib.request as request
import pandas as pd
import numpy as np
import zipfile
import joblib
import json
import mlflow
import dagshub
from mlproject import logger
from pathlib import Path
from mlproject.utils.common import get_size
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from mlproject.entities.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
            url = self.config.source_URL,
            filename = self.config.local_data_file)
            logger.info(f'{filename} downlod with following information: \n{headers}')
        else:
            logger.info(f"File already exixts of the size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    
class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)-> bool:
        validation_status = None

        data = pd.read_csv(self.config.unzip_data_dir)
        all_cols = list(data.columns)

        all_schema = self.config.all_schema.keys()

            
        for col in all_cols:
            if col not in all_schema:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
            else:
                validation_status = True
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")

        return validation_status
    


        
class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.columns_to_drop = ['Unnamed: 0', 'flight', 'duration','days_left']
        self.target_column = 'price'

    def train_test_spliting(self):
            data = pd.read_csv(self.config.data_path)
            
            # Drop specified columns
            data.drop(self.columns_to_drop, axis=1, inplace=True)
            logger.info(f"Dropped columns: {self.columns_to_drop}")
            
            train, test = train_test_split(data, test_size=0.25, random_state=42)

            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Split data into training and test sets")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")

            return train, test
            

    
    def preprocess_features(self, train, test):
            # Identify numerical and categorical columns
            numerical_columns = train.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = train.select_dtypes(include=["object", "category"]).columns

            # Exclude the target column from numerical columns
            if self.target_column in numerical_columns:
                numerical_columns = numerical_columns.drop(self.target_column)

            logger.info(f"Numerical columns: {list(numerical_columns)}")
            logger.info(f"Categorical columns: {list(categorical_columns)}")

            # Preprocessing pipelines
            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ],
                remainder="passthrough"
            )

            # Separate features and target
            train_x = train.drop(columns=[self.target_column])
            test_x = test.drop(columns=[self.target_column])
            train_y = train[self.target_column]
            test_y = test[self.target_column]

            # Fit preprocessor and transform features
            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            # Ensure target is 2D array
            train_y = train_y.values.reshape(-1, 1)
            test_y = test_y.values.reshape(-1, 1)

            # Combine processed features with target
            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            # Save preprocessor
            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")

            # Save processed data
            np.save(os.path.join(self.config.root_dir, "train_processed.npy"), train_combined)
            np.save(os.path.join(self.config.root_dir, "test_processed.npy"), test_combined)

            logger.info("Preprocessed train and test data saved successfully.")
            return train_processed, test_processed

    
    


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Validate file paths
        if not os.path.exists(self.config.train_data_path):
            raise FileNotFoundError(f"Training data file not found at {self.config.train_data_path}")
        if not os.path.exists(self.config.test_data_path):
            raise FileNotFoundError(f"Testing data file not found at {self.config.test_data_path}")

        # Load the data
        train_data = np.load(self.config.train_data_path, allow_pickle=True)
        test_data = np.load(self.config.test_data_path, allow_pickle=True)

        logger.info(f"Loaded train data: type={type(train_data)}, shape={train_data.shape}")
        logger.info(f"Loaded test data: type={type(test_data)}, shape={test_data.shape}")

        # Split features and target
        train_x = train_data[:, :-1]  # All columns except the last one
        train_y = train_data[:, -1]   # Only the last column
        test_x = test_data[:, :-1]    # All columns except the last one
        test_y = test_data[:, -1]     # Only the last column

        logger.info(f"Training data shape: X={train_x.shape}, y={train_y.shape}")
        logger.info(f"Testing data shape: X={test_x.shape}, y={test_y.shape}")

        # Train the model
        logger.info("Initializing XGBRegressor...")
        regressor = XGBRegressor(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=42
        )
        logger.info("Training the model...")
        regressor.fit(train_x, train_y)

        # Save the trained model
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(regressor, model_path)
        logger.info(f"Model saved successfully at {model_path}")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Initialize MLflow tracking
        os.environ['MLFLOW_TRACKING_USERNAME'] = "JavithNaseem-J"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "f86a04d73e27422c7a114f3f76b93fbc81a114fe"
        
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="Laptop-Price-Prediction")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/Laptop-Price-Prediction.mlflow")
        mlflow.set_experiment("Laptop Price Prediction")

    def evaluate(self):
        try:
            # Validate file paths
            if not os.path.exists(self.config.test_data_path):
                raise FileNotFoundError(f"Test data file not found at {self.config.test_data_path}")
            if not os.path.exists(self.config.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {self.config.preprocessor_path}")
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found at {self.config.model_path}")

            with mlflow.start_run():
                # Set tags for the run
                mlflow.set_tag("model_type", "LGBMRegressor")
                mlflow.set_tag("evaluation_stage", "testing")

                # Load preprocessor and model
                logger.info("Loading preprocessor and model...")
                preprocessor = joblib.load(self.config.preprocessor_path)
                model = joblib.load(self.config.model_path)

                # Log model parameters
                mlflow.log_params(self.config.all_params)

                # Load test data
                logger.info(f"Loading test data from {self.config.test_data_path}...")
                test_data = pd.read_csv(self.config.test_data_path)

                # Extract target column
                if self.config.target_column not in test_data.columns:
                    raise KeyError(f"Target column '{self.config.target_column}' not found in test data")

                test_y = test_data[self.config.target_column]
                test_x = test_data.drop(columns=[self.config.target_column])

                logger.info(f"Test data shape: X={test_x.shape}, y={test_y.shape}")

                # Preprocess test features
                logger.info("Preprocessing test features...")
                test_x_transformed = preprocessor.transform(test_x)

                # Make predictions
                logger.info("Making predictions on the test data...")
                predictions = model.predict(test_x_transformed)

                # Calculate and log metrics
                logger.info("Evaluating model performance...")
                mse = mean_squared_error(test_y, predictions)
                mae = mean_absolute_error(test_y, predictions)
                r2 = r2_score(test_y, predictions)
                adjusted_r2 = 1 - (1 - r2) * ((test_x.shape[0] - 1) / (test_x.shape[0] - test_x.shape[1] - 1))

                logger.info(f"Model Evaluation Metrics:\nMSE: {mse}\nMAE: {mae}\nR2: {r2}\nAdjusted R2: {adjusted_r2}")

                # Save the evaluation metrics
                metrics_path = os.path.join(self.config.root_dir, "metrics.json")
                metrics = {
                    "mean_squared_error": mse,
                    "mean_absolute_error": mae,
                    "r2_score": r2,
                    "adjusted_r2_score": adjusted_r2
                }
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Log model with signature
                signature = mlflow.models.infer_signature(
                    test_x_transformed, predictions
                )
                mlflow.sklearn.log_model(
                    model,
                    "wine_quality_model",
                    signature=signature,
                    registered_model_name="Flight Fare Prediction"
                )

                logger.info(f"Model Evaluation Metrics:\naccuracy: {metrics['accuracy']}\n"
                          f"precision: {metrics['precision']}\nrecall: {metrics['recall']}\n"
                          f"f1: {metrics['f1']}")

                # Save metrics locally
                metrics_path = os.path.join(self.config.root_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Evaluation metrics saved at {metrics_path}")

                return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")