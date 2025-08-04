import pandas as pd
from FareFinder.entities.config_entity import DataValidationConfig


class Validation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            
            try:
                all_schema = list(self.config.all_schema.keys())
            except AttributeError:
                all_schema = list(self.config.all_schema)
            
            validation_status = True
            
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False
                    break
            
            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
            
        except Exception as e:
            raise e