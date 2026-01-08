import os
import numpy as np
import pandas as pd
import joblib
import pickle
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

load_dotenv()

# Ortak değişkenler
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "zodiacdb")
USER_COLLECTION = os.getenv("USER_COLLECTION", "user_zodiac_data")
PRED_COLLECTION = os.getenv("PRED_COLLECTION", "model_predictions")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
user_data_col = db[USER_COLLECTION]
predictions_col = db[PRED_COLLECTION]

FEATURES = [
    "risk_taking", "emotionality", "leadership", "organization", "sociability", "patience", "flexibility", "conflict_avoidance", "social_initiative", "control_need"
]

# Kullanıcı verisini kaydet

def save_data_mongo(data):
    doc = {f: int(data[i]) for i, f in enumerate(FEATURES)}
    doc["zodiac"] = data[-1]
    user_data_col.insert_one(doc)

# Tahmin ve gerçek sonucu kaydet

def save_predictions_and_truth_mongo(user_features, preds, true_zodiac):
    doc = {f: int(user_features[i]) for i, f in enumerate(FEATURES)}
    doc.update({
        "logistic_pred": preds.get("logistic"),
        "rf_pred": preds.get("rf"),
        "knn_pred": preds.get("knn"),
        "true_zodiac": true_zodiac,
        "timestamp": datetime.now().isoformat()
    })
    predictions_col.insert_one(doc)

# Model raporu için verileri oku

def get_model_report_mongo():
    result = {}
    for model in ["logistic", "rf", "knn"]:
        pred_col = f"{model}_pred"
        total = predictions_col.count_documents({pred_col: {"$exists": True}})
        correct = predictions_col.count_documents({pred_col: {"$exists": True}, "$expr": {"$eq": [f"${pred_col}", "$true_zodiac"]}})
        incorrect = total - correct
        acc = correct / total if total > 0 else 0
        result[model] = {"accuracy": acc, "total": total, "correct": int(correct), "incorrect": int(incorrect)}
    return result

# Tüm burçlar
ZODIACS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

# Label encoder (sabit sıralama için)
label_encoder = LabelEncoder()
label_encoder.fit(ZODIACS)

# MongoDB'den tüm kullanıcı verilerini çek
def get_all_user_data_mongo():
    data = list(user_data_col.find({}, {"_id": 0}))
    if not data:
        return pd.DataFrame(columns=FEATURES + ["zodiac"])
    return pd.DataFrame(data)

# Sentetik veri üretimi (600 adet - her burçtan 50)
def generate_synthetic_data():
    import random
    profiles = {
        "Aries":       [(4,5), (1,3), (4,5), (1,3), (4,5), (1,2), (3,5), (1,2), (5,5), (4,5)],
        "Taurus":      [(1,3), (2,4), (1,3), (4,5), (2,3), (4,5), (1,2), (3,5), (1,2), (4,5)],
        "Gemini":      [(3,5), (1,3), (2,4), (1,2), (5,5), (1,2), (5,5), (2,4), (5,5), (1,3)],
        "Cancer":      [(1,3), (5,5), (2,4), (3,5), (2,4), (4,5), (2,3), (4,5), (2,3), (3,5)],
        "Leo":         [(4,5), (3,5), (5,5), (2,4), (5,5), (2,3), (3,4), (1,2), (5,5), (5,5)],
        "Virgo":       [(1,2), (1,3), (2,4), (5,5), (2,4), (4,5), (1,2), (3,5), (2,4), (4,5)],
        "Libra":       [(2,4), (3,5), (2,4), (2,4), (5,5), (3,4), (4,5), (5,5), (4,5), (1,3)],
        "Scorpio":     [(4,5), (4,5), (4,5), (3,5), (1,3), (3,5), (1,2), (1,2), (2,4), (5,5)],
        "Sagittarius": [(5,5), (2,4), (3,5), (1,2), (5,5), (1,2), (5,5), (2,4), (5,5), (1,2)],
        "Capricorn":   [(2,4), (1,2), (5,5), (5,5), (2,3), (4,5), (1,2), (2,4), (3,5), (5,5)],
        "Aquarius":    [(3,5), (1,3), (3,5), (2,4), (4,5), (3,4), (5,5), (2,4), (4,5), (2,4)],
        "Pisces":      [(1,3), (5,5), (1,3), (1,3), (3,5), (4,5), (4,5), (5,5), (2,3), (1,3)]
    }
    all_data = []
    for z in ZODIACS:
        for _ in range(50):
            ranges = profiles[z]
            sample = [random.randint(r[0], r[1]) for r in ranges]
            sample.append(z)
            all_data.append(sample)
    return pd.DataFrame(all_data, columns=FEATURES + ["zodiac"])

# Modelleri ağırlıklandırılmış şekilde eğit
def train_models_weighted():
    # Sentetik veri (600 adet)
    synthetic_df = generate_synthetic_data()
    # MongoDB'den gerçek kullanıcı verileri
    real_df = get_all_user_data_mongo()
    
    if real_df.empty:
        # Sadece sentetik veriyle eğit
        full_df = synthetic_df
        weights = np.ones(len(synthetic_df))
    else:
        # Birleştir
        full_df = pd.concat([synthetic_df, real_df], ignore_index=True)
        # Ağırlıklar: sentetik=1, gerçek=10
        weights = np.array([1.0] * len(synthetic_df) + [10.0] * len(real_df))
    
    X = full_df[FEATURES].values
    y = label_encoder.transform(full_df["zodiac"])
    
    # SGDClassifier (Logistic Regression benzeri, artımlı öğrenme destekli)
    sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42, warm_start=True)
    sgd_model.fit(X, y, sample_weight=weights)
    joblib.dump(sgd_model, "log_model.joblib")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y, sample_weight=weights)
    joblib.dump(rf_model, "rf_model.joblib")
    
    # KNN (ağırlık desteklemez, tüm veriyle eğitilir)
    knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn_model.fit(X, y)
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    
    return sgd_model, rf_model, knn_model

# Yeni kullanıcı verisiyle modeli artımlı güncelle
def incremental_update_model(user_features, true_zodiac):
    # Modelleri yeniden eğit (MongoDB'deki tüm verilerle + sentetik)
    train_models_weighted()

# Label encoder'ı dışarıya aç
def get_label_encoder():
    return label_encoder
