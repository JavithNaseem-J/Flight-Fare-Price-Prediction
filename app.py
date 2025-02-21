from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import pandas as pd
from src.mlproject.pipeline.predictionpipeline import PredictionPipeline

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=template_dir)
app.secret_key = 'your-secret-key-here'  # Required for session

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Validate required fields
            required_fields = ['airline', 'source_city', 'departure_time', 'stops', 
                               'arrival_time', 'destination_city', 'class']
            
            # Check for missing fields
            for field in required_fields:
                if field not in request.form or not request.form[field]:
                    return jsonify({'error': f"Please fill in the {field.replace('_', ' ')}"})
            
            # Validate departure and arrival times
            if request.form['departure_time'] == request.form['arrival_time']:
                return jsonify({'error': 'Departure time and arrival time cannot be the same'})
            
            # Validate source and destination cities
            if request.form['source_city'] == request.form['destination_city']:
                return jsonify({'error': 'Source and destination cities cannot be the same'})
            
            # Get form data from the user
            data = {field: request.form[field] for field in required_fields}
            
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([data])
            
            # Initialize the PredictionPipeline
            pipeline = PredictionPipeline()
            
            # Make a prediction
            prediction = pipeline.predict(input_df)
            
            # Round the prediction to 2 decimal places
            prediction = round(float(prediction), 2)
            
            # Store the data and prediction in session
            session['prediction_data'] = {
                'prediction': prediction,
                'input_data': data
            }
            
            # Redirect to results page
            return jsonify({'redirect': url_for('show_results')})
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)  # Log the error
            return jsonify({'error': error_msg})

@app.route('/results')
def show_results():
    # Get prediction data from session
    prediction_data = session.get('prediction_data')
    if not prediction_data:
        return redirect(url_for('homePage'))
    
    return render_template('results.html', 
                          prediction=prediction_data['prediction'], 
                          input_data=prediction_data['input_data'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)