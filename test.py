from src.FareFinder.pipeline.predictionpipeline import PredictionPipeline
import pandas as pd

sample_data = {
    'Airline': ['IndiGo'],
    'Source': ['CXB'],
    'Destination': ['CCU'],
    'Stopovers': ['Direct'],
    'Class': ['Business'],
    'Booking Source': ['Online Website'],
    'Days Before Departure': [10],
    'Arrival Time': ['Morning'],
    'Departure Time': ['Morning']
}
input_df = pd.DataFrame(sample_data)

pipeline = PredictionPipeline()

try:
    prediction = pipeline.predict(input_df)
    print("\n=== PREDICTION RESULT ===")
    print("Prediction:", prediction)
except Exception as e:
    print("Error during prediction:", str(e))