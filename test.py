from src.mlproject.pipeline.predictionpipeline import PredictionPipeline
import pandas as pd

# Sample input data
sample_data = {
    'airline': ['GO_FIRST'],
    'source_city': ['Mumbai'],
    'departure_time': ['Morning'],
    'stops': ['one'],
    'arrival_time': ['Early Morning'],
    'destination_city': ['Chennai'],
    'class': ['Business']
}
input_df = pd.DataFrame(sample_data)

# Initialize the PredictionPipeline
pipeline = PredictionPipeline()

try:
    # Make a prediction
    prediction = pipeline.predict(input_df)
    print("\n=== PREDICTION RESULT ===")
    print("Prediction:", prediction)
except Exception as e:
    print("Error during prediction:", str(e))