import logging
import os
from config import BASE_DIR

# إنشاء فولدر للـ Logs لو مش موجود
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name):
    logger = logging.getLogger(name)
    
    # عشان نمنع تكرار الـ Logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # حفظ الـ Logs في ملف
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'pipeline.log'), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # طباعة الـ Logs في الكونسول (الشاشة)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger