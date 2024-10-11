# feature_encoding.py
import pandas as pd
from logging_config import setup_logger

logger = setup_logger('feature_encoding', 'feature_encoding.log')

def one_hot_encode(data, categorical_columns):
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    logger.info(f"One-hot encoding applied. New shape: {data_encoded.shape}")
    return data_encoded
