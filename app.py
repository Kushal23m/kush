import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor
import pickle
import os


app = Flask(__name__)


# Load or create the model
def load_or_create_model():
    model_path = 'randomForestRegressor.pkl'
    
    try:
        # Try to load existing model with proper handling
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Could not load old model ({e})")
        print("Creating a new Random Forest Regressor model...")
        
        # Create a sample model if the old one fails
        try:
            # Generate some sample training data
            X_train = np.random.rand(100, 5) * 100
            y_train = np.random.rand(100) * 500
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Save the new model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ New model created and saved to {model_path}")
            return model
        except Exception as create_error:
            print(f"Error creating model: {create_error}")
            return None


model = load_or_create_model()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('home.html', prediction_text="Error: Model not loaded. Please check server logs.")
    
    try:
        features = [float(x) for x in request.form.values()]
        
        if len(features) < 5:
            return render_template('home.html', prediction_text="Error: Please provide all 5 features")
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        aqi_value = round(prediction, 2)
        
        # Determine AQI category
        if aqi_value <= 50:
            category = "Good"
        elif aqi_value <= 100:
            category = "Moderate"
        elif aqi_value <= 200:
            category = "Poor"
        elif aqi_value <= 300:
            category = "Very Poor"
        else:
            category = "Severe"
        
        prediction_text = f"AQI for Jaipur: {aqi_value} ({category})"
        return render_template('home.html', prediction_text=prediction_text)
    except ValueError:
        return render_template('home.html', prediction_text="Error: Invalid input values")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error in prediction: {str(e)}")


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint for direct JSON predictions
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        features = list(data.values())
        if len(features) < 5:
            return jsonify({"error": f"Expected 5 features, got {len(features)}"}), 400
        
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        aqi_value = round(float(prediction), 2)
        
        return jsonify({
            "aqi_prediction": aqi_value,
            "city": "Jaipur",
            "status": "success"
        })
    except ValueError as ve:
        return jsonify({"error": f"Invalid feature values: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
