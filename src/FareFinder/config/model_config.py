from FareFinder.constants.paths import *
from FareFinder.utils.helpers import read_yaml, create_directories
from FareFinder.entities.config_entity import ModelTrainerConfig, ModelEvaluationConfig


class ModelConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_model_training_config(self) -> ModelTrainerConfig:
            config = self.config.trainer
            params = self.params.GradientBoostingRegressor
            schema = self.schema
            random_search_params = params.random_search
            cv_params = params.cross_validation

            create_directories([config.root_dir])

            model_trainer_config = ModelTrainerConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                target_column=schema.target_column.name,
                random_search_params=random_search_params, 
                cv_folds=cv_params.cv_folds,            
                scoring=cv_params.scoring,             
                n_jobs=cv_params.n_jobs,
                n_iter=cv_params.n_iter          
            )
            
            return model_trainer_config
        

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.evaluation
        params = self.params.GradientBoostingRegressor
        schema = self.schema.target_column

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_raw_data=config.test_raw_data,
            model_path=config.model_path,
            all_params=params,
            metric_file_path=config.metric_file_path,
            preprocessor_path=config.preprocessor_path,
            target_column=schema.name,
        )
        return model_evaluation_config