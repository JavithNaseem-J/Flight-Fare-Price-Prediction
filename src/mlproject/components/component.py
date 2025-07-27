import os
import urllib.request as request
import mlflow.sklearn
import pandas as pd
import numpy as np
import zipfile
import joblib
import json
import gdown # type: ignore
import mlflow
import mlflow.sklearn
import dagshub
from mlproject import logger
from pathlib import Path
from mlproject.utils.common import get_size
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from mlproject.entities.config_entity import DataIngestionConfig, DataValidationConfig,DataCleaningConfig, DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(id=self.config.source_id, output=self.config.local_data_file, quiet=False)
            print(f"[INFO] Downloaded file: {self.config.local_data_file}")
        else:
            print(f"[INFO] File already exists: {self.config.local_data_file} ({get_size(Path(self.config.local_data_file))})")

    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
            print(f"[INFO] Extracted files to: {unzip_dir}")

    
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            
            try:
                all_schema = list(self.config.all_schema.keys())
            except AttributeError:
                all_schema = list(self.config.all_schema)
            
            validation_status = True
            
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    break
            
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
            
        except Exception as e:
            raise e
    

class DataCleaning:
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
                status_data = json.load(f)
            validation_status = status_data.get("Validation status", False)
            logger.info(f"Data validation status: {validation_status}")
            return validation_status
        except Exception as e:
            logger.error(f"Error reading validation status: {e}")
            return False
    
    def clean_data(self):
        validation_status = self.check_status()
        
        if not validation_status:
            logger.error("Data validation failed. Skipping data cleaning.")
        
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

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config.cleaned_file), exist_ok=True)
        df.to_csv(self.config.cleaned_file, index=False)
        logger.info("Data cleaning completed successfully")


        
class DataTransformation:
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
            raise e
    

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="FareFinder")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/FareFinder.mlflow")
        mlflow.set_experiment("Fare-Price-Prediction")

    def train(self):
        # Validate file paths
        if not os.path.exists(self.config.train_data_path):
            logger.error(f"Train preprocessed file not found at: {self.config.train_data_path}")
            raise FileNotFoundError("Train preprocessed file not found")
        if not os.path.exists(self.config.test_data_path):
            logger.error(f"Test preprocessed file not found at: {self.config.test_data_path}")
            raise FileNotFoundError("Test preprocessed file not found")

        # Load preprocessed data
        train_data = np.load(self.config.train_data_path, allow_pickle=True)
        test_data = np.load(self.config.test_data_path, allow_pickle=True)

        logger.info(f'Loaded train and test data')
        logger.info(f'Train data shape: {train_data.shape}')
        logger.info(f'Test data shape: {test_data.shape}')

        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]

    
        mlflow.sklearn.autolog()  
        with mlflow.start_run(run_name="RandomizedSearchCV_Tuning"):
            mlflow.set_tag("run_type", "hyperparameter_tuning")
            mlflow.set_tag("model", "GradientBoostingRegressor")

            logger.info('Initializing Randomized Search')

            gradient_model = GradientBoostingRegressor()

            param_dist = self.config.random_search_params

            logger.info('>>>>>>>>>> ......Performing Randomized Search - this may take some time...... <<<<<<<<<')


            random_search = RandomizedSearchCV(
                estimator=gradient_model,
                param_distributions=param_dist,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                scoring= self.config.scoring,
                verbose=1,
                n_jobs=self.config.n_jobs,
                return_train_score=True
            )
            random_search.fit(train_x, train_y)

            for i, (params, mean_score, std_score) in enumerate(zip(
                    random_search.cv_results_["params"],
                    random_search.cv_results_["mean_test_score"],
                    random_search.cv_results_["std_test_score"])):
                
                with mlflow.start_run(nested=True, run_name=f"Trial_{i+1}"):
                    mlflow.set_tag("trial_number", i + 1)
                    mlflow.log_params(params)
                    mlflow.log_metric("mean_r2_score", mean_score)
                    mlflow.log_metric("std_r2_score", std_score)  
                    logger.info(f"Trial {i+1}: params={params}, mean_r2_score={mean_score:.4f}, std_r2_score={std_score:.4f}")


            best_model = random_search.best_estimator_
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="gradient_model",
                registered_model_name="Flight Fare Prediction"
            )
            logger.info("Best model logged to MLflow")

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            joblib.dump(random_search, model_path)
            logger.info(f'Model saved locally at {model_path}')




class ModelEvaluation:
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
