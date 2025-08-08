from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_mail import Mail, Message
import joblib
import pandas as pd
import logging
import json



app = Flask(__name__)
CORS(app)
model = joblib.load('model2.pkl')
logging.basicConfig(level=logging.DEBUG)

# --- Flask-Mail SMTP config ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Change as needed
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'aparnashekadar123@gmail.com'  # Set your email
app.config['MAIL_PASSWORD'] = 'your_app_password'     # Set your app password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'
mail = Mail(app)
# Add a root endpoint for a welcome message
# Configure MongoDB connection
@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    recipient = data.get('recipient')
    subject = data.get('subject', 'Health Risk Assessment Results')
    body = data.get('body', '')
    if not recipient or not body:
        return jsonify({'status': 'error', 'message': 'Recipient and body required'}), 400
    try:
        msg = Message(subject=subject, recipients=[recipient], body=body)
        mail.send(msg)
        return jsonify({'status': 'success', 'message': f'Email sent to {recipient}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Load feature names used during training
with open('feature_names.json', 'r') as f:
    expected_cols = json.load(f)

# Add a root endpoint for a welcome message
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Health Risk Prediction API! Available endpoints: /store, /data, /predict'})

# Configure MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/riskdb"
mongo = PyMongo(app)

model = joblib.load('model2.pkl')
logging.basicConfig(level=logging.DEBUG)

@app.route('/store', methods=['POST'])
def store_data():
    app.logger.debug('Received request at /store')
    data = request.get_json()
    app.logger.debug(f'Request JSON: {data}')
    try:
        mongo.db.risk_data.insert_one(data)
        app.logger.debug('Data committed to database successfully')
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f'Error in /store: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/data', methods=['GET'])
def get_data():
    app.logger.debug('Received request at /data')
    entries = list(mongo.db.risk_data.find({}, {'_id': 0}))
    app.logger.debug(f'Entries fetched: {entries}')
    return jsonify(entries)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': prediction})

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


