# evaluation.py
from sklearn.metrics import classification_report
from logging_config import setup_logger

logger = setup_logger('evaluation', 'evaluation.log')

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    logger.info("Model evaluation completed.")
    logger.info(f"Evaluation Report: {report}")
    return report
