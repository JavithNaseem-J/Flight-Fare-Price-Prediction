from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str
    preprocessor_path: Path


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    model_path:Path
    test_data_path:Path
    all_params:dict
    metric_file_path:Path
    preprocessor_path: Path
    target_column:str