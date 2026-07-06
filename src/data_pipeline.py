import sqlite3
import pandas as pd
import os
from config import DB_PATH, DATA_DIR
from logger import get_logger

logger = get_logger("DataPipeline")

def ingest_raw_data():
    """
    This module is responsible for loading raw CSV files into the SQLite Database.
    Note: For the current submission, the cleaned tables are already 
    persisted in the faers_2025.db via the EDA notebooks.
    """
    logger.info("📥 Validating Database Connection and Raw Data Storage...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ Database not found at {DB_PATH}. Please run the initial cleaning notebook.")
        return False
        
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        logger.info(f"✅ Database connected successfully. Found {len(tables)} tables.")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    ingest_raw_data()