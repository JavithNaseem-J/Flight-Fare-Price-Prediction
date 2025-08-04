from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_id:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    input_data_path: Path
    cleaned_file: Path
    file_status: str 
    columns_to_drop: list
    datetime_columns: list
    target_column_mapping: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str
    preprocessor_path: Path
    label_encoder: Path
    categorical_columns: list
    numerical_columns: list


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str
    random_search_params: dict
    n_iter: int     
    cv_folds: int
    scoring: str 
    n_jobs: int
    

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_raw_data: Path
    all_params: dict
    metric_file_path: Path
    preprocessor_path: Path
    target_column: str