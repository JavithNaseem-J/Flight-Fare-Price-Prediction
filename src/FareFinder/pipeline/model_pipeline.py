from FareFinder.config.model_config import ModelConfigurationManager
from FareFinder.components.model.training import Trainer
from FareFinder.components.model.evaluation import Evaluation
from FareFinder import logger

class ModelPipeline:
    def __init__(self):
        pass

    def main(self):

        config = ModelConfigurationManager()


        model_trainer_config = config.get_model_training_config()
        model_trainer_config = Trainer(config=model_trainer_config)
        model_trainer_config.train()


        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = Evaluation(config=model_evaluation_config)
        metrics = model_evaluation.evaluate()