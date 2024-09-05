from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame([data])
    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)