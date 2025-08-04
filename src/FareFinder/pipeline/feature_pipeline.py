from FareFinder.config.data_config import DataConfigurationManager
from FareFinder.components.feature.cleaning import Cleaning
from FareFinder.components.feature.transform import Transformation
from FareFinder import logger

class FeaturePipeline:
    def __init__(self):
        pass

    def main(self):
        config = DataConfigurationManager()

        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning = Cleaning(config=data_cleaning_config)
        data_cleaning.clean_data()


        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Transformation(config=data_transformation_config)
        train, test = data_transformation.train_test_splitting()
        train_processed, test_processed = data_transformation.preprocess_features(train, test)


