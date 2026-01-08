import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pickle
from mongo_utils import save_data_mongo, save_predictions_and_truth_mongo, get_model_report_mongo, incremental_update_model, get_label_encoder
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

@app.route('/report')
def serve_report():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'report.html')

FEATURES = [
    "risk_taking", "emotionality", "leadership", "organization", "sociability", "patience", "flexibility", "conflict_avoidance", "social_initiative", "control_need"
]
RF_MODEL_FILE = "rf_model.joblib"
KNN_MODEL_FILE = "knn_model.pkl"
LOG_MODEL_FILE = "log_model.joblib"

@app.route('/')
def index():
    return render_template('zodiac_survey.html')

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json
    user_features = [int(data.get(f)) for f in FEATURES]
    preds = {}
    le = get_label_encoder()
    # Logistic Regression
    if os.path.exists(LOG_MODEL_FILE):
        log_model = joblib.load(LOG_MODEL_FILE)
        pred_idx = log_model.predict([user_features])[0]
        preds['logistic'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['logistic'] = None
    # Random Forest
    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        pred_idx = rf_model.predict([user_features])[0]
        preds['rf'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['rf'] = None
    # KNN
    if os.path.exists(KNN_MODEL_FILE):
        with open(KNN_MODEL_FILE, "rb") as f:
            knn_model = pickle.load(f)
        pred_idx = knn_model.predict([user_features])[0]
        preds['knn'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['knn'] = None
    return jsonify(preds)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_features = [int(data.get(f)) for f in FEATURES]
    true_zodiac = data.get("true_zodiac")
    preds = {}
    le = get_label_encoder()
    if os.path.exists(LOG_MODEL_FILE):
        log_model = joblib.load(LOG_MODEL_FILE)
        pred_idx = log_model.predict([user_features])[0]
        preds['logistic'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['logistic'] = None
    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        pred_idx = rf_model.predict([user_features])[0]
        preds['rf'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['rf'] = None
    if os.path.exists(KNN_MODEL_FILE):
        with open(KNN_MODEL_FILE, "rb") as f:
            knn_model = pickle.load(f)
        pred_idx = knn_model.predict([user_features])[0]
        preds['knn'] = le.inverse_transform([pred_idx])[0]
    else:
        preds['knn'] = None
    # MongoDB'ye kaydet
    save_data_mongo(user_features + [true_zodiac])
    save_predictions_and_truth_mongo(user_features, preds, true_zodiac)
    # Modeli ağırlıklandırılmış şekilde güncelle
    incremental_update_model(user_features, true_zodiac)
    return jsonify({"status": "success"})

@app.route('/model_report')
def model_report():
    result = get_model_report_mongo()
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
