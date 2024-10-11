# feature_scaling.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logging_config import setup_logger

logger = setup_logger('feature_scaling', 'feature_scaling.log')

def scale_features(data, target_column='target'):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Features scaled successfully. Shape: {X_scaled.shape}")

    return X_scaled, y
