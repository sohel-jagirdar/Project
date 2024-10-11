# data_cleaning.py
from logging_config import setup_logger

logger = setup_logger('data_cleaning', 'data_cleaning.log')

def clean_data(data):
    cleaned_data = data.drop_duplicates()
    logger.info(f"Data cleaned successfully. Shape: {cleaned_data.shape}")
    return cleaned_data
