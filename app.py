import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="Drug Safety Intelligence System", layout="wide", initial_sidebar_state="collapsed")

# 2. Custom CSS for Sleek Dark/Light Mode UI
st.markdown("""
    <style>
        /* General Spacing */
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h3 { font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 600; }
        
        /* Modern Primary Button */
        .stButton>button { 
            background-color: #2563eb; 
            color: #ffffff; 
            border-radius: 8px; 
            border: none; 
            padding: 0.75rem 2rem; 
            font-weight: 600; 
            transition: all 0.3s ease; 
        }
        .stButton>button:hover { background-color: #1d4ed8; color: #ffffff; }

        /* Rich Result Boxes (Adapts perfectly to Dark Mode) */
        .result-box-high { 
            padding: 1.5rem; border-radius: 8px; 
            background-color: rgba(239, 68, 68, 0.1); 
            border-left: 4px solid #ef4444; color: #f87171; 
        }
        .result-box-standard { 
            padding: 1.5rem; border-radius: 8px; 
            background-color: rgba(34, 197, 94, 0.1); 
            border-left: 4px solid #22c55e; color: #4ade80; 
        }
        .reaction-box { 
            padding: 1.5rem; border-radius: 8px; 
            background-color: rgba(245, 158, 11, 0.1); 
            border-left: 4px solid #f59e0b; color: #fbbf24; 
        }
        
        .result-title { font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem; display: block; }
        .result-subtitle { font-size: 0.95rem; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# 3. Path Configuration & Data Loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

@st.cache_resource
def load_artifacts():
    try:
        sev_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_severity_model.pkl'))
        sev_encoders = joblib.load(os.path.join(MODEL_DIR, 'severity_encoders.pkl'))
        reac_model = joblib.load(os.path.join(MODEL_DIR, 'matrix_b_ensemble_TUNED.pkl'))
        reac_le = joblib.load(os.path.join(MODEL_DIR, 'le_matrix_b.pkl'))
        reac_oe = joblib.load(os.path.join(MODEL_DIR, 'oe_matrix_b.pkl'))
        return sev_model, sev_encoders, reac_model, reac_le, reac_oe
    except Exception as e:
        st.error(f"System Error: Unable to load models. Details: {e}")
        return None, None, None, None, None

@st.cache_data
def load_signals():
    try:
        return pd.read_csv(os.path.join(REPORT_DIR, 'safety_signals_report.csv'))
    except:
        return pd.DataFrame(columns=["drug", "adverse_event", "report_count"])

sev_model, sev_encoders, reac_model, reac_le, reac_oe = load_artifacts()
df_signals = load_signals()

# 4. Main Header
st.markdown("<h1>Drug Safety Intelligence System</h1>", unsafe_allow_html=True)

# 5. UI Tabs Setup
tab1, tab2 = st.tabs(["Patient Risk Analysis", "Global Safety Signals"])

# ==========================================
# TAB 1: PATIENT RISK ANALYSIS
# ==========================================
with tab1:
    st.markdown("<p style='color: #94a3b8; margin-bottom: 2rem;'>Enter patient and clinical parameters to generate an integrated risk assessment.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<h3>Patient Demographics</h3>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=0, max_value=120, value=50, step=1)
        wt = st.number_input("Weight (KG)", min_value=1.0, max_value=250.0, value=70.0, step=1.0)
        sex = st.selectbox("Sex", ["M", "F", "UNK"])

    with col2:
        st.markdown("<h3>Clinical Data</h3>", unsafe_allow_html=True)
        primary_suspect_drug = st.text_input("Primary Suspect Drug", value="aspirin").lower()
        primary_indication = st.text_input("Primary Indication", value="pain").lower()
        ps_route = st.selectbox("Route of Administration", ["ORAL", "INTRAVENOUS", "SUBCUTANEOUS", "UNKNOWN"])
        num_drugs = st.number_input("Concurrent Drugs", min_value=1, max_value=30, value=1, step=1)
        therapy_duration = st.number_input("Therapy Duration (Days)", min_value=1, max_value=2000, value=30, step=1)

    with col3:
        st.markdown("<h3>Report Metadata</h3>", unsafe_allow_html=True)
        rept_cod = st.selectbox("Report Type", ["EXP", "PER", "UNK"], index=0)
        rpsr_cod = st.selectbox("Report Source", ["SS", "C", "PS", "UNK"], index=0)
        occp_cod = st.selectbox("Reporter Occupation", ["MD", "PH", "LW", "CN", "OT", "UNK"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Logic
    if st.button("Generate Risk Assessment", width='stretch'):
        if sev_model and reac_model:
            input_dict = {
                'age': age, 'wt': wt, 'num_drugs': num_drugs, 'therapy_duration': therapy_duration, 'num_indications': 1,
                'primary_suspect_drug': primary_suspect_drug, 'ps_route': ps_route, 'primary_indication': primary_indication,
                'rept_cod': rept_cod, 'occp_cod': occp_cod, 'rpsr_cod': rpsr_cod, 'sex': sex
            }
            df_input = pd.DataFrame([input_dict])

            # Severity Prediction
            df_sev = df_input.copy()
            for col in ['primary_suspect_drug', 'ps_route', 'primary_indication']:
                val = df_sev[col].iloc[0]
                df_sev[col + '_encoded'] = sev_encoders[col].get(val, sev_encoders['overall_mean']) if col in sev_encoders else sev_encoders['overall_mean']
                df_sev.drop(columns=[col], inplace=True)
                
            df_sev = pd.get_dummies(df_sev, columns=['sex'])
            for c in set(sev_encoders['train_columns']) - set(df_sev.columns): df_sev[c] = 0
            df_sev = df_sev[sev_encoders['train_columns']]
            
            sev_pred = sev_model.predict(df_sev)[0]
            sev_prob = sev_model.predict_proba(df_sev)[0][1]

            # Reaction Prediction
            df_reac = df_input.copy()
            cat_cols = ['primary_suspect_drug', 'ps_route', 'rept_cod', 'occp_cod', 'rpsr_cod', 'primary_indication', 'sex']
            df_reac[cat_cols] = df_reac[cat_cols].astype(str)
            df_reac[cat_cols] = reac_oe.transform(df_reac[cat_cols])
            
            # 🛠️ THE FIX: Reordering columns to match training exactly!
            expected_cols = ['age', 'wt', 'sex', 'occp_cod', 'rept_cod', 'primary_suspect_drug', 'ps_route', 'num_drugs', 'therapy_duration', 'rpsr_cod', 'num_indications', 'primary_indication']
            df_reac = df_reac[expected_cols]
            
            reac_probs = reac_model.predict_proba(df_reac)[0]
            top_3_idx = reac_probs.argsort()[-3:][::-1]

            # Display Results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3>Integrated Assessment Results</h3>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 1], gap="large")
            
            with res_col1:
                if sev_pred == 1:
                    st.markdown(f"""
                    <div class='result-box-high'>
                        <span class='result-title'>High Risk Profile Detected</span>
                        Calculated Probability: {sev_prob * 100:.1f}%<br>
                        <span class='result-subtitle'>Patient profile indicates a strong correlation with severe clinical outcomes.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box-standard'>
                        <span class='result-title'>Standard Risk Profile</span>
                        Calculated Probability: {sev_prob * 100:.1f}%<br>
                        <span class='result-subtitle'>No significant correlation with severe clinical outcomes detected.</span>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                reactions_html = "<div class='reaction-box'><span class='result-title'>Predicted Adverse Events</span><ul style='margin-top: 10px; margin-bottom: 0;'>"
                for idx in top_3_idx:
                    reaction_name = reac_le.inverse_transform([idx])[0].title()
                    confidence = reac_probs[idx] * 100
                    reactions_html += f"<li><strong>{reaction_name}</strong> (Confidence: {confidence:.1f}%)</li>"
                reactions_html += "</ul></div>"
                
                st.markdown(reactions_html, unsafe_allow_html=True)

# ==========================================
# TAB 2: GLOBAL SAFETY SIGNALS
# ==========================================
with tab2:
    st.markdown("<p style='color: #94a3b8; margin-bottom: 2rem;'>Historical drug-adverse event pairs requiring clinical investigation.</p>", unsafe_allow_html=True)
    
    if not df_signals.empty:
        search_drug = st.text_input("Search Database by Drug Name:", "").lower()
        display_df = df_signals[df_signals['drug'].str.lower().str.contains(search_drug)] if search_drug else df_signals
            
        st.dataframe(
            display_df.style.background_gradient(subset=['report_count'], cmap='Blues'),
            width='stretch',
            height=500
        )
    else:
        st.warning("No Safety Signals report found. Please execute the data pipeline.")