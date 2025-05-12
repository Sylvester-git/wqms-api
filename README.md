# Water Quality Monitoring API

A Flask-based REST API for real-time water quality monitoring, alerting, and prediction. Integrates MongoDB for data storage, Firebase Cloud Messaging and Slack for notifications, and a machine learning model for water safety prediction.

---

## Features

- **User Registration & Login**: Register and authenticate users with FCM tokens for push notifications.
- **Sensor Data Ingestion**: Accepts water quality sensor data (pH, turbidity, temperature, TDS).
- **Data Storage**: Persists all data in MongoDB.
- **Real-Time Alerts**: Sends alerts via Firebase and Slack if water quality is unsafe.
- **Machine Learning Prediction**: Predicts water safety using a Random Forest classifier.
- **RESTful API**: Provides endpoints for data submission, retrieval, and prediction.

---

## Environment Setup

1. **Clone the repository** and install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

2. **Configure environment variables** in a `.env` file:

    ``` bash
    MONGO_URI=your_mongodb_uri
    FIREBASE_SERVICE_ACCOUNT_KEY=your_firebase_service_account_json
    FIREBASE_PROJECT_ID=...
    FIREBASE_PRIVATE_KEY_ID=...
    FIREBASE_PRIVATE_KEY=...
    FIREBASE_CLIENT_EMAIL=...
    FIREBASE_CLIENT_ID=...
    FIREBASE_AUTH_URI=...
    FIREBASE_TOKEN_URI=...
    FIREBASE_AUTH_PROVIDER_CERT_URL=...
    FIREBASE_CLIENT_CERT_URL=...
    FIREBASE_UNIVERSE_DOMAIN=...
    SLACK_WEBHOOK_URL=your_slack_webhook_url
    PORT=5000
    ```

3. **Run the application**:

    ```sh
    python app.py
    ```

---

## API Endpoints

### `GET /`

- **Description**: Health check endpoint.
- **Response**: `"Welcome to the Water Quality Monitoring API!"`

---

### `POST /api/register`

- **Description**: Register a new user.
- **Request Body**:

    ```json
    {
      "username": "user1",
      "password": "pass",
      "fcmtoken": "firebase_token"
    }
    ```

- **Response**: User info or error.

---

### `POST /api/login`

- **Description**: Authenticate a user.
- **Request Body**:

    ```json
    {
      "username": "user1",
      "password": "pass"
    }
    ```

- **Response**: User info or error.

---

### `POST /api/sensor_data`

- **Description**: Submit new sensor data.
- **Request Body**:

    ```json
    {
      "pH": 7.2,
      "turbidity": 2.5,
      "temp": 25.0,
      "tds": 500
    }
    ```

- **Response**: Status, prediction, confidence, and alerts.

---

### `GET /api/sensor_data`

- **Description**: Retrieve the 50 most recent sensor data records.
- **Response**: List of sensor data.

---

### `GET /api/predict`

- **Description**: Get prediction and alerts for the latest sensor data.
- **Response**: Water quality status, confidence, and alerts.

---

## Machine Learning Model

- **Algorithm**: Random Forest Classifier (`sklearn`)
- **Features**: pH, turbidity, temperature, TDS
- **Label**: Unsafe (1) if any parameter is out of WHO range, else Safe (0)
- **Training**: Retrained every 10 new records (if at least 10 records exist)
- **Prediction**: Returns "Safe" or "Unsafe" with confidence score

---

## Notifications

- **Firebase**: Sends push notifications to the first registered user's FCM token.
- **Slack**: Sends alerts to a configured Slack channel via webhook.
- **Trigger**: Alerts are sent if any parameter is out of the safe range.

---

## WHO Water Quality Standards

| Parameter   | Safe Range                |
|-------------|---------------------------|
| pH          | 6.5 - 8.5                 |
| Turbidity   | ≤ 5.0 NTU                 |
| Temperature | ≤ 30.0 °C                 |
| TDS         | ≤ 1000.0 ppm              |

---

## Code Structure

- **Flask App Initialization**: Sets up Flask, logging, and loads environment variables.
- **MongoDB Connection**: Connects to MongoDB (cloud/local) and initializes collections.
- **Firebase Initialization**: Loads credentials and initializes Firebase Admin SDK.
- **Notification Functions**: Functions to send alerts via Firebase and Slack.
- **Water Quality Check**: Checks sensor data against WHO standards and triggers notifications.
- **Model Training**: Trains a Random Forest model on historical data.
- **API Endpoints**: Implements all REST endpoints for registration, login, data submission, retrieval, and prediction.
- **Shutdown Hook**: Closes MongoDB connection gracefully on shutdown.

---

## Shutdown & Cleanup

- The application registers a shutdown hook to close the MongoDB connection when the app exits.

---

## License

This project is licensed under the MIT License.

---

**Note:**  

- Ensure all environment variables are set correctly for MongoDB, Firebase, and Slack integration.
- The API is designed for demonstration and prototyping; for production, enhance security (e.g., password hashing, authentication, validation).
