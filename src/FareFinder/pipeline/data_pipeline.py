from FareFinder.config.data_config import DataConfigurationManager
from FareFinder.components.data.ingestion import Ingestion
from FareFinder.components.data.validation import Validation
from FareFinder import logger
import os

class DataPipeline:
    def __init__(self):
        pass
    def main(self):
        config = DataConfigurationManager()

        
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Ingestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


        data_validation_config = config.get_data_validation_config()
        data_validation = Validation(config=data_validation_config)
        data_validation.validate_all_columns()



