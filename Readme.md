<div align="center">

# 💊 Drug Safety Intelligence System

### AI-Powered Pharmacovigilance & Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-FF9900?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[🌟 Overview](#-overview) • [✨ Features](#-features) • [📸 Screenshots](#-screenshots) • [🛠️ Tech Stack](#️-tech-stack) • [📦 Installation](#-installation) • [👥 Team](#-team)

---

<img src="https://img.icons8.com/fluency/256/000000/pill.png" alt="System Logo" width="120"/>

### **Transform raw clinical data into proactive patient safety insights** ✨

<br/>

[![Web](https://img.shields.io/badge/Web_App-4285F4?style=flat&logo=google-chrome&logoColor=white)](/)
[![API](https://img.shields.io/badge/REST_API-009688?style=flat&logo=fastapi&logoColor=white)](/)
[![Data](https://img.shields.io/badge/FDA_FAERS-2025-blue?style=flat&logo=database&logoColor=white)](/)

</div>

---

## 🌟 Overview

The **Drug Safety Intelligence System** is an end-to-end, AI-powered pharmacovigilance platform designed to analyze adverse drug events and proactively predict patient harm severity and specific reactions. Utilizing real-world clinical data, the system assists healthcare professionals and clinical researchers in early signal detection and risk assessment.

> 🏥 **FDA FAERS Integration** - Built on massive real-world relational databases (Year 2025) including DEMO, DRUG, REAC, THER, and OUTC.

### 🎯 What Makes This System Special?

<table>
<tr>
<td width="50%">

#### 🧠 Advanced ML Models
Utilizes highly optimized **XGBoost** models for both Binary Severity Prediction (ROC-AUC: 85.08%) and Multi-class Reaction Forecasting.

#### ⚖️ Class Imbalance Handling
Programmatically tackles real-world clinical data imbalance using strategic downsampling and rigorous Target/Mean Encoding strategies.

#### 📊 Global Safety Signals
Mines integrated clinical databases to identify and flag statistically significant drug-adverse event pairs automatically.

</td>
<td width="50%">

#### ☁️ Cloud-Optimized Architecture
Refactored from a massive 2GB ensemble to a streamlined **~53MB `joblib`-compressed model**, perfect for strict memory environments.

#### 🚀 Dual-Deployment
Features both a beautiful, interactive **Streamlit Dashboard** for end-users and a robust **FastAPI Backend** for system integrations.

#### 🛠️ Automated SQL Pipeline
Constructs complex feature engineering matrices directly using SQL Common Table Expressions (CTEs) without overwhelming RAM.

</td>
</tr>
</table>

---

## ✨ Features

### 📱 Core Modules

<table>
<tr>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/96/000000/medical-doctor.png" width="48"/>
<br/><strong>🔬 Patient Risk Analysis</strong>
<br/><sub>Calculates real-time severity probabilities based on patient profiles</sub>
</td>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/96/000000/doctors-bag.png" width="48"/>
<br/><strong>🔮 Reaction Forecasting</strong>
<br/><sub>Predicts the top 3 adverse events a patient might experience</sub>
</td>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/96/000000/combo-chart.png" width="48"/>
<br/><strong>🚨 Signal Detection</strong>
<br/><sub>Interactive database exploring historical drug-reaction reports</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="https://img.icons8.com/fluency/96/000000/api-settings.png" width="48"/>
<br/><strong>⚡ Oriva PV API</strong>
<br/><sub>FastAPI endpoints for seamless frontend/external integration</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/96/000000/data-configuration.png" width="48"/>
<br/><strong>⚙️ Dual Pipeline</strong>
<br/><sub>Distinct, rigorous preprocessing strategies for different models</sub>
</td>
<td align="center">
<img src="https://img.icons8.com/fluency/96/000000/system-report.png" width="48"/>
<br/><strong>📈 Deep EDA Reports</strong>
<br/><sub>High-res clinical pattern visualizations and data drift checks</sub>
</td>
</tr>
</table>

### 🤖 Machine Learning Capabilities

| Task | Algorithm | Optimization | Performance (Test Set) |
|:-----|:----------|:-------------|:-----------------------|
| 🔴 **Severity Prediction** | `XGBClassifier` (Binary) | `scale_pos_weight=1.40`, `tree_method='hist'` | **ROC-AUC: 85.08%**<br>Recall: 77.00% |
| 💊 **Reaction Forecasting** | `XGBClassifier` (Multi-class) | `max_depth=10`, `compress=9` (~53MB footprint) | **Accuracy: 72.85%**<br>(Across 20 targets) |

---

## 📸 Screenshots

<div align="center">

### 💻 Live System Dashboard

<table width="100%">
<tr>
<td align="center" width="50%"><strong>Patient Risk Analysis (Matrix A & B)</strong></td>
<td align="center" width="50%"><strong>Global Safety Signals Database</strong></td>
</tr>
<tr>
<td align="center"><img src="reports/figures/dashboard_preview.png" width="100%" alt="Dashboard Preview"/></td>
<td align="center"><img src="reports/figures/safety_signals_tab.png" width="100%" alt="Safety Signals Tab"/></td>
</tr>
</table>

### 📊 Deep Exploratory Data Analysis (EDA)

<table width="100%">
<tr>
<td align="center" width="33%"><strong>Demographics (Age/Weight)</strong></td>
<td align="center" width="33%"><strong>Target Imbalance</strong></td>
<td align="center" width="33%"><strong>Top 10 Suspect Drugs</strong></td>
</tr>
<tr>
<td align="center"><img src="reports/figures/age_weight_distribution.png" width="100%"/></td>
<td align="center"><img src="reports/figures/target_imbalance.png" width="100%"/></td>
<td align="center"><img src="reports/figures/top_10_drugs.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><strong>Polypharmacy Impact</strong></td>
<td align="center"><strong>Age by Gender</strong></td>
<td align="center"><strong>Feature Importance</strong></td>
</tr>
<tr>
<td align="center"><img src="reports/figures/polypharmacy.png" width="100%"/></td>
<td align="center"><img src="reports/figures/age_by_gender.png" width="100%"/></td>
<td align="center"><img src="reports/figures/feature_importance.png" width="100%"/></td>
</tr>
<tr>
<td align="center" colspan="3"><strong>Therapy Duration Analysis</strong></td>
</tr>
<tr>
<td align="center" colspan="3"><img src="reports/figures/therapy_duration_log.png" width="80%"/></td>
</tr>
<tr>
<td align="center" colspan="3"><strong>Temporal Stability (Data Drift)</strong></td>
</tr>
<tr>
<td align="center" colspan="3"><img src="reports/figures/data_drift_check.png" width="80%"/></td>
</tr>
</table>

</div>

---

## 🛠️ Tech Stack

<div align="center">

### Data Processing & Pipeline

| Technology | Purpose |
|:----------:|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core Programming Language (3.9+) |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Clinical Data Manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Mathematical Operations |
| ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white) | Relational Database Engine |

### Machine Learning

| Technology | Purpose |
|:----------:|---------|
| ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Preprocessing, Resampling, Metrics |
| ![XGBoost](https://img.shields.io/badge/XGBoost-FF9900?style=flat&logo=xgboost&logoColor=white) | Tree-based Classification Algorithms |
| ![Optuna](https://img.shields.io/badge/Optuna-252933?style=flat&logo=python&logoColor=white) | Hyperparameter Tuning |
| ![Joblib](https://img.shields.io/badge/Joblib-000000?style=flat) | Advanced Model Compression |

### Deployment & Backend

| Technology | Purpose |
|:----------:|---------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | High-performance RESTful API |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interactive Web Dashboard |
| ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white) | CI/CD Pipeline (Linting) |

</div>

---

## 📦 Installation

<details>
<summary><strong>📋 Prerequisites</strong></summary>

- ✅ Python 3.9 or higher
- ✅ Git installed
- ✅ Raw FDA FAERS Dataset (Not included in repo due to size)

</details>

### 🚀 Quick Start

```bash
# 1️⃣ Clone the repository
git clone [https://github.com/GoldenBoy13420/Drug-Safety-Intelligence-System.git](https://github.com/GoldenBoy13420/Drug-Safety-Intelligence-System.git)
cd Drug-Safety-Intelligence-System

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit Dashboard
streamlit run app.py

# 4️⃣ Or Run the FastAPI Backend
uvicorn api:app --reload
```

To rebuild the matrices and retrain the models from scratch:

```bash
python main.py
```

This single entry point will sequentially run:

1. Database Connection Validation
2. Deep EDA (`eda.py`)
3. Safety Signal Detection (`signal_detector.py`)
4. Matrix A & B Construction (`preprocessing.py`)
5. XGBoost Model Training & Evaluation

---

## 📁 Project Structure

```text
DRUG-SAFETY-INTELLIGENCE-SYSTEM/
│
├── 📂 data/                   # Raw FDA data & SQLite DB (Ignored in Git)
├── 📂 logs/                   # System execution logs (pipeline.log)
├── 📂 models/                 # Compressed XGBoost .pkl models & Encoders
├── 📂 notebooks/              # Data integration & experimental notebooks
├── 📂 reports/
│   ├── 📂 figures/            # High-res EDA plots & Dashboard Screenshots
│   └── 📄 safety_signals_report.csv
│
├── 💻 src/                    # Core Modules
│   ├── config.py              # System paths and constants
│   ├── data_pipeline.py       # Database validation
│   ├── eda.py                 # Visual analysis generator
│   ├── preprocessing.py       # SQL CTEs and Matrix Building
│   ├── model_severity.py      # Binary Classification Pipeline
│   ├── model_adverse_event.py # Multi-class Classification Pipeline
│   └── signal_detector.py     # Pharmacovigilance Mining
│
├── ⚡ api.py                  # FastAPI endpoints
├── 🌐 app.py                  # Streamlit UI dashboard
├── 🚀 main.py                 # Full Pipeline orchestrator
└── 📄 requirements.txt        # Cloud dependencies
```

---

## 👥 Team

<div align="center">

<table>
<tr>

<td align="center">
<img src="https://github.com/GoldenBoy13420.png" width="80" style="border-radius: 50%"/>
<br/><strong>Mahmoud Abdelrauf</strong>
<br/><sub>🎯 AI Engineer / Lead</sub>
<br/>
<a href="https://github.com/GoldenBoy13420">
<img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/>
</a>
</td>

<td align="center">
<img src="https://github.com/AhmedElabd.png" width="80" style="border-radius: 50%"/>
<br/><strong>Ahmed Elabd</strong>
<br/><sub>💻 Data Scientist</sub>
<br/>
<a href="https://github.com/AhmedElabd">
<img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/>
</a>
</td>

<td align="center">
<img src="https://github.com/Batran7.png" width="80" style="border-radius: 50%"/>
<br/><strong>Abdelrhman Batran</strong>
<br/><sub>💻 Data Scientist</sub>
<br/>
<a href="https://github.com/Batran7">
<img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/>
</a>
</td>

</tr>
</table>

</div>

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Support

If you find this intelligence system helpful, please give it a star!
