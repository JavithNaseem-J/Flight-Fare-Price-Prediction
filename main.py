import argparse
from FareFinder import logger
from FareFinder.pipeline.data_pipeline import DataPipeline
from FareFinder.pipeline.feature_pipeline import FeaturePipeline
from FareFinder.pipeline.model_pipeline import ModelPipeline


def run_stage(stage_name):
    logger.info(f">>>>>> Stage {stage_name} started <<<<<<")

    try:
        if stage_name == "data_pipeline":
            stage = DataPipeline()
            stage.main()

        elif stage_name == "feature_pipeline":
            stage = FeaturePipeline()
            stage.main()

        elif stage_name == "model_pipeline":
            stage = ModelPipeline()
            stage.main()

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration or data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific pipeline stage.")
    parser.add_argument("--stage", help="Name of the stage to run")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
    else:
        stages = [
            "data_pipeline",
            "feature_pipeline",
            "model_pipeline",
        ]
        for stage in stages:
            run_stage(stage)
