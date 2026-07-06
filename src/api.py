from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from src.config import MODEL_DIR

app = FastAPI(title="Oriva PV AI API", version="1.0", description="Predict Adverse Event Severity and Reaction")

# تحميل موديلات Severity (Matrix A)
severity_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_severity_model.pkl'))
severity_encoders = joblib.load(os.path.join(MODEL_DIR, 'severity_encoders.pkl'))

# تحميل موديلات Reaction (Matrix B)
reaction_model = joblib.load(os.path.join(MODEL_DIR, 'matrix_b_ensemble_TUNED.pkl'))
le_target = joblib.load(os.path.join(MODEL_DIR, 'le_matrix_b.pkl'))
oe_features = joblib.load(os.path.join(MODEL_DIR, 'oe_matrix_b.pkl'))

# هيكل البيانات الموحد للمريض
class PatientData(BaseModel):
    age: float
    wt: float
    sex: str
    num_drugs: int
    therapy_duration: float
    primary_suspect_drug: str
    ps_route: str
    primary_indication: str
    rept_cod: str = "Unknown"
    occp_cod: str = "Unknown"
    rpsr_cod: str = "Unknown"

@app.get("/")
def home():
    return {"message": "Welcome to Oriva Pharmacovigilance API 🚀"}

@app.post("/predict_severity")
def predict_severity(patient: PatientData):
    input_data = pd.DataFrame([patient.dict()])
    target_cols = ['primary_suspect_drug', 'ps_route', 'primary_indication']
    
    for col in target_cols:
        if col in severity_encoders:
            input_data[col + '_encoded'] = severity_encoders[col].get(input_data[col][0], severity_encoders['overall_mean'])
        input_data.drop(columns=[col], inplace=True)
    
    input_data = pd.get_dummies(input_data, columns=['sex'])
    missing_cols = set(severity_encoders['train_columns']) - set(input_data.columns)
    for c in missing_cols:
        input_data[c] = 0
    input_data = input_data[severity_encoders['train_columns']]
    
    prediction = severity_model.predict(input_data)[0]
    probability = severity_model.predict_proba(input_data)[0][1]
    
    return {
        "is_severe": bool(prediction),
        "risk_probability": round(float(probability), 4),
        "status": "High Risk ⚠️" if prediction == 1 else "Standard Risk ✅"
    }

@app.post("/predict_reaction")
def predict_reaction(patient: PatientData):
    input_data = pd.DataFrame([patient.dict()])
    
    cat_cols = ['primary_suspect_drug', 'ps_route', 'rept_cod', 'occp_cod', 'rpsr_cod', 'primary_indication', 'sex']
    num_cols = ['age', 'wt', 'num_drugs', 'therapy_duration', 'num_indications']
    
    input_data[cat_cols] = input_data[cat_cols].fillna('Unknown').astype(str)
    input_data[num_cols] = input_data[num_cols].fillna(-1)
    
    # Ordinal Encoding
    input_data[cat_cols] = oe_features.transform(input_data[cat_cols])
    
    # توقع العرض الجانبي
    pred_encoded = reaction_model.predict(input_data)[0]
    predicted_reaction = le_target.inverse_transform([pred_encoded])[0]
    
    # سحب أعلى 3 احتمالات
    probabilities = reaction_model.predict_proba(input_data)[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_reactions = [
        {"reaction": le_target.inverse_transform([i])[0], "probability": round(float(probabilities[i]), 4)}
        for i in top_3_indices
    ]
    
    return {
        "predicted_reaction": predicted_reaction,
        "top_3_risks": top_3_reactions
    }