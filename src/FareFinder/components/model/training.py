import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import dagshub
from pathlib import Path
from FareFinder.utils.logging import logger
from sklearn.ensemble import GradientBoostingRegressor
from FareFinder.entities.config_entity import ModelTrainerConfig
from sklearn.model_selection import RandomizedSearchCV


class Trainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        try:
            dagshub.init(repo_owner="JavithNaseem-J", repo_name="FareFinder")
            mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/FareFinder.mlflow")
            mlflow.set_experiment("Fare-Price-Prediction")
            self.mlflow_enabled = True
            logger.info("MLflow tracking initialized successfully")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            logger.info("Continuing without MLflow tracking")
            self.mlflow_enabled = False

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

        try:
            if self.mlflow_enabled:
  
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

                    try:
                        mlflow.log_params(random_search.best_params_)
                        mlflow.log_metric("best_score", random_search.best_score_)
                        logger.info(f"Best parameters: {random_search.best_params_}")
                        logger.info(f"Best score: {random_search.best_score_}")
                    except Exception as e:
                        logger.warning(f"Error logging parameters to MLflow: {e}")

                    best_model = random_search.best_estimator_
                    
                    try:
                        mlflow.sklearn.log_model(
                            sk_model=best_model,
                            artifact_path="gradient_model",
                            registered_model_name="Flight Fare Prediction"
                        )
                        logger.info("Best model logged to MLflow")
                    except Exception as e:
                        logger.warning(f"Error logging model to MLflow: {e}")
                        logger.info("Model will still be saved locally")
            else:
                raise Exception("MLflow not enabled, using fallback")

        except Exception as e:
            logger.warning(f"MLflow error: {e}")
            logger.info("Falling back to training without MLflow logging...")
            
            # Fallback training without MLflow
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
            
            logger.info(f"Best parameters: {random_search.best_params_}")
            logger.info(f"Best score: {random_search.best_score_}")

        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(random_search, model_path)
        logger.info(f'Model saved locally at {model_path}')