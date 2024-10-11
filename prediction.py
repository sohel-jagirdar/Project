# prediction.py
import joblib
from sklearn.metrics import accuracy_score
from logging_config import setup_logger

logger = setup_logger('prediction', 'prediction.log')

def load_model(filename='best_model.pkl'):
    model = joblib.load(filename)
    logger.info(f"Model loaded successfully from {filename}")
    return model

def make_prediction(model, X_new):
    predictions = model.predict(X_new)

    logger.info(f"Predictions made successfully.")
    return predictions
