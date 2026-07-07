import pandas as pd
import sqlite3
import joblib
import os
import xgboost as xgb
# تم إزالة LightGBM و RandomForest لأننا مش هنحتاجهم في الـ Production
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import resample, shuffle
from src.config import DB_PATH, MATRIX_B, MODEL_DIR

def train_adverse_event_model():
    print("🚀 Loading Matrix B for Lightweight Production Training...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {MATRIX_B}", conn)
    conn.close()

    top_20_reactions = df['target_reaction'].value_counts().nlargest(20).index.tolist()
    df['target_mapped'] = df['target_reaction'].apply(lambda x: x if x in top_20_reactions else 'Other')

    le_target = LabelEncoder()
    df['target_encoded'] = le_target.fit_transform(df['target_mapped'])
    num_classes = len(le_target.classes_)

    train_mask = df['is_test_set'] == 0
    test_mask = df['is_test_set'] == 1

    drop_cols = ['target_reaction', 'target_mapped', 'target_encoded', 'is_test_set', 'primaryid']
    X_train = df[train_mask].drop(columns=drop_cols).copy()
    y_train = df[train_mask]['target_encoded'].copy()
    X_test = df[test_mask].drop(columns=drop_cols).copy()
    y_test = df[test_mask]['target_encoded'].copy()

    # Balancing Data
    other_class_val = le_target.transform(['Other'])[0]
    X_majority = X_train[y_train == other_class_val]
    y_majority = y_train[y_train == other_class_val]
    X_minority = X_train[y_train != other_class_val]
    y_minority = y_train[y_train != other_class_val]

    X_maj_down, y_maj_down = resample(X_majority, y_majority, replace=False, n_samples=len(X_minority), random_state=42)
    X_train = pd.concat([X_maj_down, X_minority])
    y_train = pd.concat([y_maj_down, y_minority])
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    cat_cols = ['primary_suspect_drug', 'ps_route', 'rept_cod', 'occp_cod', 'rpsr_cod', 'primary_indication', 'sex']
    num_cols = ['age', 'wt', 'num_drugs', 'therapy_duration', 'num_indications']

    X_train[cat_cols] = X_train[cat_cols].fillna('Unknown').astype(str)
    X_test[cat_cols] = X_test[cat_cols].fillna('Unknown').astype(str)
    X_train[num_cols] = X_train[num_cols].fillna(-1)
    X_test[num_cols] = X_test[num_cols].fillna(-1)

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])

    print("🚀 Training Final Lightweight XGBoost Model for Deployment...")
    # تم الإبقاء على XGBoost فقط عشان يكون سريع ومساحته صغيرة جداً
    production_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=num_classes, tree_method='hist',
        max_depth=10, learning_rate=0.137, n_estimators=400, random_state=42, n_jobs=-1
    )

    production_xgb.fit(X_train, y_train)
    
    print("💾 Saving Models...")
    # تم تغيير اسم الموديل عشان يعكس إنه XGBoost خفيف
    joblib.dump(production_xgb, os.path.join(MODEL_DIR, 'xgb_matrix_b_model.pkl'))
    joblib.dump(le_target, os.path.join(MODEL_DIR, 'le_matrix_b.pkl'))
    joblib.dump(oe, os.path.join(MODEL_DIR, 'oe_matrix_b.pkl'))
    print("✅ Training Pipeline Completed!")

if __name__ == "__main__":
    train_adverse_event_model()