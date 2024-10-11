# data_loading.py
from sklearn.datasets import load_iris
import pandas as pd
from logging_config import setup_logger

logger = setup_logger('data_loading', 'data_loading.log')

def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    return data
