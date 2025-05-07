from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, messaging
import requests
import atexit

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection setup
def get_mongo_client():
    mongo_uri = os.getenv('MONGO_URI')
    if mongo_uri:
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=60000)
            client.server_info()
            logger.info("Connected to online MongoDB successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to online MongoDB: {e}")
            exit(1)
    else:
        try:
            client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=60000)
            client.server_info()
            logger.info("Connected to local MongoDB successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to local MongoDB: {e}")
            exit(1)

# Initialize MongoDB
mongo_client = get_mongo_client()
db = mongo_client['water_quality_db']
sensor_data_collection = db['sensor_data']
user_data_collection = db['users']
alert_collection = db['alerts']


# Initialize Firebase Admin SDK (uncomment if using)
# cred = credentials.Certificate('firebase-adminsdk.json')
# firebase_admin.initialize_app(cred)

# WHO safe parameter thresholds
PH_MIN = 6.5
PH_MAX = 8.5
TURBIDITY_MAX = 5.0  # NTU
TEMP_MAX = 30.0      # °C (practical upper limit)
TDS_MAX = 1000.0     # ppm (WHO guideline for drinking water)

# Send Firebase notification
def send_firebase_notification(message):
    fcm_token = os.getenv('FCM_TOKEN')
    if not fcm_token:
        logger.warning("FCM_TOKEN not set in .env")
        return
    notification = messaging.Message(
        notification=messaging.Notification(
            title='Water Quality Alert',
            body=message,
        ),
        token=fcm_token,
    )
    try:
        response = messaging.send(notification)
        logger.info(f"Firebase notification sent: {response}")
    except Exception as e:
        logger.error(f"Failed to send Firebase notification: {e}")

# Send Slack notification
def send_slack_notification(message):
    slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if not slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set in .env")
        return
    payload = {
        "text": f"Water Quality Alert: {message}"
    }
    try:
        response = requests.post(slack_webhook_url, json=payload)
        response.raise_for_status()
        logger.info("Slack notification sent successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to send Slack notification: {e}")

# Check water quality against WHO standards and send notifications
def check_water_quality_and_notify(data):
    pH = data['pH']
    turbidity = data['turbidity']
    temp = data['temp']
    tds = data['tds']
    alerts = []

    if pH < PH_MIN:
        alert = f"Low pH detected: {pH} (below {PH_MIN})"
        alerts.append(alert)
        send_firebase_notification(alert)
        send_slack_notification(alert)
    elif pH > PH_MAX:
        alert = f"High pH detected: {pH} (above {PH_MAX})"
        alerts.append(alert)
        send_firebase_notification(alert)
        send_slack_notification(alert)
    
    if turbidity > TURBIDITY_MAX:
        alert = f"High turbidity detected: {turbidity} NTU (above {TURBIDITY_MAX} NTU)"
        alerts.append(alert)
        send_firebase_notification(alert)
        send_slack_notification(alert)
    
    if temp > TEMP_MAX:
        alert = f"High temperature detected: {temp}°C (above {TEMP_MAX}°C)"
        alerts.append(alert)
        send_firebase_notification(alert)
        send_slack_notification(alert)
    
    if tds > TDS_MAX:
        alert = f"High TDS detected: {tds} ppm (above {TDS_MAX} ppm)"
        alerts.append(alert)
        send_firebase_notification(alert)
        send_slack_notification(alert)

    return alerts

# Train Random Forest model
def train_model():
    data = list(sensor_data_collection.find().limit(100))
    if len(data) < 10:  # Minimum data for training
        logger.warning("Insufficient data for training")
        return None
    df = pd.DataFrame(data)
    X = df[['pH', 'turbidity', 'temp', 'tds']]
    y = [1 if (row['pH'] < PH_MIN or row['pH'] > PH_MAX or 
               row['turbidity'] > TURBIDITY_MAX or 
               row['temp'] > TEMP_MAX or 
               row['tds'] > TDS_MAX) 
         else 0 for _, row in df.iterrows()]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()


@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Water Quality Monitoring API!"

@app.route("/api/register", methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data recived")
            return jsonify({"error": "No data provided"}), 400
        required_fields = ['username', 'password', 'fcmtoken']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            logger.warning(f"Missing fields: {missing}")
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        username = str(data['username'])
        password = str(data['password'])
        fcmtoken = str(data['fcmtoken'])

        #Does user exist
        user_data = user_data_collection.find({"username": username}).limit(1)
        user = list(user_data)
        if (len(user) > 0):
            return jsonify({"error": "User already exist"}), 400
        else:
            user_record = {
            "username": username,
            "password": password,
            "fcmtoken": fcmtoken,
            "created_at": datetime.utcnow()
            }
            # Insert data into MongoDB
            result = user_data_collection.insert_one(user_record)
            logger.info(f"Data inserted with ID: {result.inserted_id}")
            return jsonify({"status": True, "data": {
                "username": user_record['username'],
                "fcmtoken": user_record['fcmtoken'],
                "password": user_record['password']
            }})
    except ValueError as ve:
        logger.error(f"Invalid data format: {ve}")
        return jsonify({"error": "Invalid data format (values must be numbers)"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500
        

@app.route("/api/login", methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data recived")
            return jsonify({"error": "No data provided"}), 400
        required_fields = ['username', 'password']
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            logger.warning(f"Missing fields: {missing}")
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        username = str(data['username'])
        user_data = list(user_data_collection.find({"username": username}).limit(1))
        logger.info(len(user_data))
        if (len(user_data) > 0 ):
            logger.info(user_data)
            return jsonify({"data": {
                "username": user_data[0]['username'],
                "fcmtoken": user_data[0]['fcmtoken'],
                "password": user_data[0]['password']
            }}), 200
        else:
            logger.info(f"User not found")
            return jsonify({"error": "User not found"}), 400
    except ValueError as ve:
        logger.error(f"Invalid data format: {ve}")
        return jsonify({"error": "Invalid data format (values must be numbers)"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500
         

# POST endpoint to receive sensor data
@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["pH", "turbidity", "temp", "tds"]
        if not all(field in data for field in required_fields):
            missing = [field for field in required_fields if field not in data]
            logger.warning(f"Missing fields: {missing}")
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        pH = float(data["pH"])
        turbidity = float(data["turbidity"])
        temp = float(data["temp"])
        tds = float(data["tds"])

        sensor_record = {
            "pH": pH,
            "turbidity": turbidity,
            "temp": temp,
            "tds": tds,
            "timestamp": datetime.utcnow()
        }

        # Insert data into MongoDB
        result = sensor_data_collection.insert_one(sensor_record)
        logger.info(f"Data inserted with ID: {result.inserted_id}")

        # Check water quality and send notifications
        alerts = check_water_quality_and_notify(sensor_record)
        if (alerts.count > 0):
            alert_record = {
            "alerts": alerts,
            "timestamp": datetime.utcnow(),
            }
            result = alert_collection.insert_one(alert_record)

        # Retrain model periodically
        global model
        if model is None or len(list(sensor_data_collection.find())) % 10 == 0:  # Retrain every 10 records
            model = train_model()

        # Predict with Random Forest
        if model:
            X = np.array([[pH, turbidity, temp, tds]])
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            status = "Unsafe" if prediction == 1 else "Safe"
            confidence = float(max(prediction_proba)) * 100
        else:
            status = "Unknown"
            confidence = 0.0

        return jsonify({
            "message": "Data received and stored successfully",
            "id": str(result.inserted_id),
            "status": status,
            "confidence": confidence,
            "alerts": alerts if alerts else "All parameters within safe limits"
        }), 201

    except ValueError as ve:
        logger.error(f"Invalid data format: {ve}")
        return jsonify({"error": "Invalid data format (values must be numbers)"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

# GET endpoint to retrieve sensor data
@app.route('/api/sensor_data', methods=['GET'])
def get_sensor_data():
    try:
        data = list(sensor_data_collection.find().sort("timestamp", -1).limit(50))
        for record in data:
            record['_id'] = str(record['_id'])
            record['timestamp'] = record['timestamp'].isoformat()
        logger.info("Retrieved sensor data successfully")
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return jsonify({"error": "Internal server error"}), 500

# GET endpoint for ML prediction and water quality status
@app.route('/api/predict', methods=['GET'])
def predict_water_quality():
    try:
        latest_data = sensor_data_collection.find().sort("timestamp", -1).limit(1)
        latest_data = list(latest_data)[0]
        
        # Check WHO standards and notify
        alerts = check_water_quality_and_notify(latest_data)

        # ML prediction
        if model is None:
            logger.warning("Model not trained yet")
            status = "Unknown"
            confidence = 0.0
        else:
            X = np.array([[latest_data['pH'], latest_data['turbidity'], latest_data['temp'], latest_data['tds']]])
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            status = "Unsafe" if prediction == 1 else "Safe"
            confidence = float(max(prediction_proba)) * 100

        response = {
            "pH": latest_data['pH'],
            "turbidity": latest_data['turbidity'],
            "temp": latest_data['temp'],
            "tds": latest_data['tds'],
            "status": status,
            "confidence": confidence,
            "alerts": alerts if alerts else "All parameters within safe limits",
            "timestamp": latest_data['timestamp'].isoformat()
        }
        logger.info(f"Prediction: {status} with {confidence}% confidence")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": f"Error in prediction: {e}"}), 500

# Function to close MongoDB connection on app shutdown
def close_mongo_connection():
    if mongo_client is not None:
        mongo_client.close()
        logger.info("MongoDB connection closed on application shutdown")

# Register shutdown hook
atexit.register(close_mongo_connection)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))