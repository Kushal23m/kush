# AQI Prediction Web App

This repository contains a simple Flask application that predicts the Air Quality Index (AQI) for Jaipur using a pre-trained machine learning model (Random Forest Regressor). The app provides both a web interface and a REST API for predictions.

## Features

- Web form to enter meteorological parameters (temperature, humidity, pressure, wind speed, precipitation)
- API endpoint (`/predict_api`) for JSON requests
- Automatic model loading with compatibility handling and fallback training
- AQI category classification (Good, Moderate, Poor, Very Poor, Severe)
- Responsive, styled HTML interface

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 tested)
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kushal23m/kush.git
   cd kush
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000` to use the web interface.

### Using the API

Send a POST request with JSON body containing feature values:

```bash
curl -X POST http://127.0.0.1:5000/predict_api \
     -H "Content-Type: application/json" \
     -d '{"temp":30,"humidity":60,"pressure":1012,"wind":10,"precip":0}'
```

A successful response:

```json
{
  "aqi_prediction": 123.45,
  "city": "Jaipur",
  "status": "success"
}
```

## Model

The application attempts to load `randomForestRegressor.pkl`. If the file is missing or incompatible, a new dummy model is trained on random data and saved. Replace this with your own trained model for production use.

## Development

- `app.py` contains the Flask routes and model logic
- `templates/home.html` contains the form and result presentation

## License

MIT License

## Acknowledgements

Built by `Kushal23m` as a demonstration of a simple ML-powered Flask web app.