

import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import pickle



app = Flask(__name__)
CORS(app)


DATA_FILE = "user_zodiac_data.csv"
FEATURES = [
    "risk_taking", "emotionality", "leadership", "organization", "sociability", "patience", "flexibility", "conflict_avoidance", "social_initiative", "control_need"
]
RF_MODEL_FILE = "rf_model.joblib"
KNN_MODEL_FILE = "knn_model.pkl"
LOG_MODEL_FILE = "log_model.joblib"
# /predict_all endpoint: returns predictions from all models
@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json
    user_features = [
        int(data.get("risk_taking")),
        int(data.get("emotionality")),
        int(data.get("leadership")),
        int(data.get("organization")),
        int(data.get("sociability")),
        int(data.get("patience")),
        int(data.get("flexibility")),
        int(data.get("conflict_avoidance")),
        int(data.get("social_initiative")),
        int(data.get("control_need"))
    ]
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
    # data: [risk_taking, ..., control_need, true_zodiac]
    columns = FEATURES + ["zodiac"]
    row = data[:-1] + [data[-1]]  # last is true_zodiac, saved as 'zodiac'
    df = pd.DataFrame([row], columns=columns)
    write_header = False
    if not os.path.exists(DATA_FILE):
        write_header = True
    else:
        if os.path.getsize(DATA_FILE) == 0:
            write_header = True
    df.to_csv(DATA_FILE, mode='a' if not write_header else 'w', header=write_header, index=False)

# Save model predictions and ground truth to a separate file
PREDICTIONS_FILE = "model_predictions.csv"
def save_predictions_and_truth(user_features, preds, true_zodiac):
    # user_features: list of 10 ints
    # preds: dict with keys 'logistic', 'rf', 'knn'
    # true_zodiac: str
    columns = FEATURES + ["logistic_pred", "rf_pred", "knn_pred", "true_zodiac", "timestamp"]
    row = user_features + [preds.get("logistic"), preds.get("rf"), preds.get("knn"), true_zodiac, datetime.now().isoformat()]
    df = pd.DataFrame([row], columns=columns)
    write_header = False
    if not os.path.exists(PREDICTIONS_FILE):
        write_header = True
    else:
        if os.path.getsize(PREDICTIONS_FILE) == 0:
            write_header = True
    df.to_csv(PREDICTIONS_FILE, mode='a' if not write_header else 'w', header=write_header, index=False)


# 1. Özellikleri al, tahmin yap, sonucu döndür

# /predict endpoint: supports both rf and knn
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_features = [
        int(data.get("risk_taking")),
        int(data.get("emotionality")),
        int(data.get("leadership")),
        int(data.get("organization")),
        int(data.get("sociability")),
        int(data.get("patience")),
        int(data.get("flexibility")),
        int(data.get("conflict_avoidance")),
        int(data.get("social_initiative")),
        int(data.get("control_need"))
    ]
    model_type = data.get("model_type", "rf")
    if model_type == "rf":
        if not os.path.exists(RF_MODEL_FILE):
            return jsonify({"error": "Random Forest model not found."}), 500
        model = joblib.load(RF_MODEL_FILE)
        pred = model.predict([user_features])[0]
    elif model_type == "knn":
        if not os.path.exists(KNN_MODEL_FILE):
            return jsonify({"error": "KNN model not found."}), 500
        with open(KNN_MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        pred = model.predict([user_features])[0]
    else:
        return jsonify({"error": "Unknown model type."}), 400
    return jsonify({"predicted_zodiac": pred})


# 2. Özellikler, tüm model tahminleri ve gerçek burç ile kaydet
@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    user_features = [
        int(data.get("risk_taking")),
        int(data.get("emotionality")),
        int(data.get("leadership")),
        int(data.get("organization")),
        int(data.get("sociability")),
        int(data.get("patience")),
        int(data.get("flexibility")),
        int(data.get("conflict_avoidance")),
        int(data.get("social_initiative")),
        int(data.get("control_need"))
    ]
    true_zodiac = data.get("true_zodiac")

    # Get model predictions for logging
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

    # Save to main data file (for training)
    save_data(user_features + [true_zodiac])
    # Save to predictions file (for analysis)
    save_predictions_and_truth(user_features, preds, true_zodiac)
    return jsonify({"status": "success"})



@app.route('/zodiac_survey.html')
def serve_survey():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'zodiac_survey.html')

# Serve rapor.html at /report
@app.route('/report')
def serve_report():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'report.html')

@app.route('/model_report')
def model_report():
    preds_file = "model_predictions.csv"
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

if __name__ == "__main__":
    app.run(debug=True)
