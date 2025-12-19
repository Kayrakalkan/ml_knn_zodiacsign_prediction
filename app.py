import os
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pickle

app = Flask(__name__)
CORS(app)

@app.route('/report')
def serve_report():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'report.html')

DATA_FILE = "user_zodiac_data.csv"
PREDICTIONS_FILE = "model_predictions.csv"
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
    # Logistic Regression
    if os.path.exists(LOG_MODEL_FILE):
        log_model = joblib.load(LOG_MODEL_FILE)
        preds['logistic'] = log_model.predict([user_features])[0]
    else:
        preds['logistic'] = None
    # Random Forest
    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        preds['rf'] = rf_model.predict([user_features])[0]
    else:
        preds['rf'] = None
    # KNN
    if os.path.exists(KNN_MODEL_FILE):
        with open(KNN_MODEL_FILE, "rb") as f:
            knn_model = pickle.load(f)
        preds['knn'] = knn_model.predict([user_features])[0]
    else:
        preds['knn'] = None
    return jsonify(preds)

def save_data(data):
    columns = FEATURES + ["zodiac"]
    row = data[:-1] + [data[-1]]
    df = pd.DataFrame([row], columns=columns)
    write_header = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0
    df.to_csv(DATA_FILE, mode='a' if not write_header else 'w', header=write_header, index=False)

def save_predictions_and_truth(user_features, preds, true_zodiac):
    columns = FEATURES + ["logistic_pred", "rf_pred", "knn_pred", "true_zodiac", "timestamp"]
    row = user_features + [preds.get("logistic"), preds.get("rf"), preds.get("knn"), true_zodiac, datetime.now().isoformat()]
    df = pd.DataFrame([row], columns=columns)
    write_header = not os.path.exists(PREDICTIONS_FILE) or os.path.getsize(PREDICTIONS_FILE) == 0
    df.to_csv(PREDICTIONS_FILE, mode='a' if not write_header else 'w', header=write_header, index=False)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_features = [int(data.get(f)) for f in FEATURES]
    true_zodiac = data.get("true_zodiac")
    # Get model predictions for logging
    preds = {}
    if os.path.exists(LOG_MODEL_FILE):
        log_model = joblib.load(LOG_MODEL_FILE)
        preds['logistic'] = log_model.predict([user_features])[0]
    else:
        preds['logistic'] = None
    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        preds['rf'] = rf_model.predict([user_features])[0]
    else:
        preds['rf'] = None
    if os.path.exists(KNN_MODEL_FILE):
        with open(KNN_MODEL_FILE, "rb") as f:
            knn_model = pickle.load(f)
        preds['knn'] = knn_model.predict([user_features])[0]
    else:
        preds['knn'] = None
    save_data(user_features + [true_zodiac])
    save_predictions_and_truth(user_features, preds, true_zodiac)
    return jsonify({"status": "success"})

@app.route('/model_report')
def model_report():
    preds_file = PREDICTIONS_FILE
    if not os.path.exists(preds_file) or os.path.getsize(preds_file) == 0:
        return jsonify({m: {"accuracy": 0, "total": 0, "correct": 0, "incorrect": 0} for m in ["logistic", "rf", "knn"]})
    df = pd.read_csv(preds_file)
    result = {}
    for model in ["logistic", "rf", "knn"]:
        pred_col = f"{model}_pred"
        if pred_col in df.columns and "true_zodiac" in df.columns:
            total = df.shape[0]
            correct = (df[pred_col] == df["true_zodiac"]).sum()
            incorrect = total - correct
            acc = correct / total if total > 0 else 0
            result[model] = {"accuracy": acc, "total": total, "correct": int(correct), "incorrect": int(incorrect)}
        else:
            result[model] = {"accuracy": 0, "total": 0, "correct": 0, "incorrect": 0}
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
