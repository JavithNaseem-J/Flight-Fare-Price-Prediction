from FareFinder.constants.paths import *
from FareFinder.utils.helpers import read_yaml, create_directories
from FareFinder.entities.config_entity import DataIngestionConfig, DataValidationConfig,DataCleaningConfig, DataTransformationConfig


class DataConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_id=config.source_id,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.validation
        schema = self.schema.columns
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )
        return data_validation_config
    


    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.cleaning
        schema = self.schema.cleaning

        create_directories([config.root_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data,
            cleaned_file=config.cleaned_file,
            file_status=config.file_status,
            columns_to_drop=schema.columns_to_drop,
            datetime_columns=schema.datetime_columns,
            target_column_mapping=schema.target_column_mapping
        )

        return data_cleaning_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.transformation
        schema = self.schema
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            target_column=config.target_column,
            preprocessor_path=Path(config.preprocessor_path),
            label_encoder=Path(config.label_encoder),
            categorical_columns=schema.categorical_columns,
            numerical_columns=schema.numeric_columns
        )
        
        return data_transformation_config