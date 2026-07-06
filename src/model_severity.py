import sqlite3
import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import roc_auc_score, classification_report
from src.config import DB_PATH, MATRIX_A, MODEL_DIR
from logger import get_logger

logger = get_logger("SeverityModel")

def train_severity_model():
    logger.info("🧠 Loading Data and Validating Features for Severity Model...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {MATRIX_A}", conn)
    conn.close()

    # 1. التحديد الصارم للميزات (Explicit Feature Selection)
    numeric_features = ['age', 'wt', 'num_drugs', 'therapy_duration', 'num_indications']
    categorical_features = ['primary_suspect_drug', 'ps_route', 'rept_cod', 'occp_cod', 'rpsr_cod', 'primary_indication']
    binary_features = ['sex'] # Sex needs One-Hot Encoding
    
    all_features = numeric_features + categorical_features + binary_features
    logger.info(f"📊 Features used for training ({len(all_features)}): {all_features}")

    # 2. التقسيم الزمني
    train_mask = df['is_test_set'] == 0
    test_mask = df['is_test_set'] == 1
    
    # اختيار الميزات المحددة فقط + الهدف
    X_train = df[train_mask][all_features].copy()
    y_train = df[train_mask]['is_severe']
    
    X_test = df[test_mask][all_features].copy()
    y_test = df[test_mask]['is_severe']

    # 3. Target Encoding للميزات النصية
    logger.info("⚙️ Applying Target Encoding for Categorical Features...")
    overall_mean = y_train.mean()
    encoders = {'overall_mean': overall_mean, 'numeric_features': numeric_features}

    for col in categorical_features:
        # حساب متوسط الخطورة لكل فئة
        target_means = y_train.groupby(X_train[col].fillna('UNK')).mean()
        encoders[col] = target_means
        
        # التطبيق على الداتا
        X_train[col + '_encoded'] = X_train[col].fillna('UNK').map(target_means).fillna(overall_mean)
        X_test[col + '_encoded'] = X_test[col].fillna('UNK').map(target_means).fillna(overall_mean)
        
        # مسح العمود النصي القديم
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

    # 4. One-Hot Encoding للجنس (Sex)
    X_train = pd.get_dummies(X_train, columns=['sex'], drop_first=False)
    X_test = pd.get_dummies(X_test, columns=['sex'], drop_first=False)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # حفظ ترتيب الأعمدة النهائي للـ API
    encoders['train_columns'] = X_train.columns.tolist()

    # 5. تدريب الموديل
    logger.info("🚀 Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=1.40,      
        max_depth=9,
        learning_rate=0.1,
        n_estimators=450,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        tree_method='hist',         
        enable_categorical=False,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 6. التقييم
    logger.info("🎯 Evaluating on Q4 Test Data...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_score = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"🌟 Final ROC-AUC Score: {roc_score:.4f}")
    print(classification_report(y_test, y_pred))

    # 7. الحفظ
    logger.info("💾 Saving Model and Encoders...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_severity_model.pkl'))
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'severity_encoders.pkl'))
    logger.info("✅ Severity Model Pipeline Completed Successfully!")

if __name__ == "__main__":
    train_severity_model()