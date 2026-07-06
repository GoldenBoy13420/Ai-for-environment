import sqlite3
import pandas as pd
import os
from config import DB_PATH, REPORT_DIR
from logger import get_logger

logger = get_logger("SignalDetector")

def detect_safety_signals(min_reports=50):
    logger.info("🕵️‍♂️ Starting Signal Detection for Drug-Adverse Event Pairs...")
    conn = sqlite3.connect(DB_PATH)
    
    # استعلام بيجيب الدواء والعرض الجانبي وبيعد اتكرروا مع بعض كام مرة
    query = """
    SELECT 
        d.final_drug_name as drug, 
        r.pt as adverse_event, 
        COUNT(*) as report_count
    FROM drug_clean d
    JOIN reac_clean r ON d.primaryid = r.primaryid
    WHERE d.role_cod = 'PS' 
      AND d.final_drug_name != 'none'
      AND r.pt IS NOT NULL
    GROUP BY d.final_drug_name, r.pt
    HAVING report_count >= ?
    ORDER BY report_count DESC
    """
    
    # تنفيذ الاستعلام
    df_signals = pd.read_sql_query(query, conn, params=(min_reports,))
    conn.close()
    
    # حفظ التقرير في ملف CSV للـ Evaluator
    output_path = os.path.join(REPORT_DIR, 'safety_signals_report.csv')
    df_signals.to_csv(output_path, index=False)
    
    logger.info(f"✅ Found {len(df_signals)} potential safety signals (Pairs with > {min_reports} reports).")
    logger.info(f"📄 Report saved to: {output_path}")
    
    print("\n🚨 Top 5 Detected Safety Signals:")
    print(df_signals.head())

if __name__ == "__main__":
    detect_safety_signals()