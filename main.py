import sys
import os

# إضافة فولدر src للمسارات عشان نقدر نستدعي الملفات اللي جواه
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline import ingest_raw_data
from src.eda import run_deep_eda
from src.signal_detector import detect_safety_signals
from src.preprocessing import build_severity_matrix, build_reaction_matrix
from src.model_severity import train_severity_model
from src.model_adverse_event import train_adverse_event_model

def run_full_pipeline():
    print("🚀 STARTING ORIVA END-TO-END PHARMACOVIGILANCE AI PIPELINE")
    
    print("\n➡️ STEP 1: Validating Database Connection")
    ingest_raw_data()
    
    print("\n➡️ STEP 2: Running Deep EDA & Generating Reports")
    run_deep_eda()
    
    print("\n➡️ STEP 3: Detecting Safety Signals")
    detect_safety_signals()
    
    print("\n➡️ STEP 4: Building ML Feature Matrices (Severity & Reaction)")
    build_severity_matrix()
    build_reaction_matrix()
    
    print("\n➡️ STEP 5: Training Severity Predictor (Matrix A)")
    train_severity_model()
    
    print("\n➡️ STEP 6: Training Adverse Event Predictor with Optuna (Matrix B)")
    train_adverse_event_model()
    
    print("\n✅ ALL PIPELINES EXECUTED SUCCESSFULLY! Models, logs, and figures are saved.")

if __name__ == "__main__":
    run_full_pipeline()