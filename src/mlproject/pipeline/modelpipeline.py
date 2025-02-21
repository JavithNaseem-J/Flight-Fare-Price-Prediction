from mlproject.config.config import ConfigurationManager
from src.mlproject.components.component import DataIngestion,DataValiadtion,DataTransformation,ModelTrainer,ModelEvaluation
from mlproject import logger

STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


STAGE_NAME = 'Data Validation'

class DataValidationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


STAGE_NAME = 'Data Transformation'


class DataTransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        train, test = data_transformation.train_test_spliting()

        train_processed, test_processed = data_transformation.preprocess_features(train, test)

        logger.info(f"Processed training data shape: {train_processed.shape}")
        logger.info(f"Processed testing data shape: {test_processed.shape}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")



STAGE_NAME = 'Data Model Training'


class ModelTrainerPipeline():
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except KeyError as e:
            logger.error(f"Missing key in configuration: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")



STAGE_NAME = 'Data Evaluation Training'



class ModelEvaluationPipeline():
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_config.evaluate()

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
        except KeyError as e:
            logger.error(f"Missing key in configuration: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluation()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


