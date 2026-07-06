import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import DB_PATH, REPORT_DIR
from logger import get_logger

logger = get_logger("DeepEDA")
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def run_deep_eda():
    logger.info("📊 Starting Comprehensive & Deep Exploratory Data Analysis (EDA)...")
    conn = sqlite3.connect(DB_PATH)

    # 1. قراءة الداتا الأساسية
    df_demo = pd.read_sql_query("SELECT * FROM demo_clean", conn)
    
    # ==========================================
    # 1. Deep EDA: Age & Weight Distributions (Subplots)
    # ==========================================
    logger.info("📈 Generating Age and Weight Distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(data=df_demo, x='age', bins=50, kde=True, color='#4169E1', ax=axes[0])
    axes[0].set_title('Age Distribution (Strictly Years)', weight='bold')
    axes[0].set_xlabel('Age')
    
    # فلترة الأوزان المنطقية فقط للرسم
    valid_weights = df_demo[(df_demo['wt'] > 0) & (df_demo['wt'] <= 300)]
    sns.histplot(data=valid_weights, x='wt', bins=50, kde=True, color='#2E8B57', ax=axes[1])
    axes[1].set_title('Weight Distribution (Strictly KG)', weight='bold')
    axes[1].set_xlabel('Weight (KG)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'age_weight_distribution.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. Deep EDA: Therapy Duration (Log Scale)
    # ==========================================
    logger.info("⏳ Generating Therapy Duration (Log Scale)...")
    df_ther = pd.read_sql_query("SELECT dur FROM ther_clean WHERE dur IS NOT NULL AND dur > 0", conn)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_ther, x='dur', bins=50, color='#F39C12', log_scale=True)
    plt.title('Distribution of Therapy Duration in Days (Log Scale)', weight='bold')
    plt.xlabel('Duration (Days) - Log Scale')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(FIGURES_DIR, 'therapy_duration_log.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. Demographic Distribution by Gender
    # ==========================================
    logger.info("👥 Generating Demographic Distributions by Gender...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_demo[df_demo['sex'].isin(['M', 'F'])], x='age', hue='sex', bins=50, kde=True, palette={'M': '#85C1E9', 'F': '#F1948A'})
    plt.title('Patient Age Distribution by Gender', weight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, 'age_by_gender.png'), dpi=300)
    plt.close()

    # ==========================================
    # 4. Target Imbalance (Severity)
    # ==========================================
    logger.info("⚖️ Checking Target Imbalance...")
    df_target = pd.read_sql_query("SELECT CASE WHEN outc_cod IN ('DE', 'LT', 'HO', 'DS', 'CA', 'RI') THEN 1 ELSE 0 END as is_severe FROM outc_clean", conn)
    
    target_counts = df_target['is_severe'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(target_counts, labels=['Non-Severe (0)', 'Severe (1)'], autopct='%1.2f%%', startangle=90, colors=['#2ECC71', '#E74C3C'], explode=[0, 0.1], shadow=True)
    plt.title('Target Variable Distribution (Imbalance Check)', weight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, 'target_imbalance.png'), dpi=300)
    plt.close()

    # ==========================================
    # 5. Polypharmacy Analysis
    # ==========================================
    logger.info("💊 Analyzing Polypharmacy...")
    df_poly = pd.read_sql_query("SELECT caseid, COUNT(final_drug_name) as num_drugs FROM drug_clean GROUP BY caseid", conn)

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_poly[df_poly['num_drugs'] <= 15], x='num_drugs', palette='viridis')
    plt.title('Polypharmacy: Number of Drugs per Patient', weight='bold')
    plt.xlabel('Number of Drugs Taken Concurrently')
    plt.savefig(os.path.join(FIGURES_DIR, 'polypharmacy.png'), dpi=300)
    plt.close()

    # ==========================================
    # 6. Temporal Stability (Data Drift)
    # ==========================================
    logger.info("🔄 Verifying Temporal Stability (Data Drift)...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_demo, x='age', hue='is_test_set', common_norm=False, fill=True, palette=['#3498DB', '#E67E22'], alpha=0.5)
    plt.title('Temporal Stability Check: Age Distribution (Train vs. Test)', weight='bold')
    plt.xlabel('Age (Years)')
    plt.savefig(os.path.join(FIGURES_DIR, 'data_drift_check.png'), dpi=300)
    plt.close()

    # ==========================================
    # 7. Top 10 Drugs Analysis (Deep EDA extra)
    # ==========================================
    logger.info("🏆 Extracting Top 10 Suspect Drugs...")
    df_drugs = pd.read_sql_query("SELECT final_drug_name FROM drug_clean WHERE role_cod = 'PS' AND final_drug_name != 'none'", conn)
    top_drugs = df_drugs['final_drug_name'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(y=top_drugs.index, x=top_drugs.values, palette='magma')
    plt.title('Top 10 Primary Suspect Drugs in Reports', weight='bold')
    plt.xlabel('Number of Reports')
    plt.ylabel('Drug Name')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'top_10_drugs.png'), dpi=300)
    plt.close()

    conn.close()
    logger.info(f"✅ Deep EDA completed! All 7 high-resolution plots are saved in: {FIGURES_DIR}")

if __name__ == "__main__":
    run_deep_eda()