# modeling.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from logging_config import setup_logger

logger = setup_logger('modeling', 'modeling.log')

def save_model(model, filename='best_model.pkl'):
    joblib.dump(model, filename)
    logger.info(f"Model saved successfully as {filename}")

def train_multiple_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42)
    }

    best_model = None
    best_f1 = 0
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results[name] = {"accuracy": accuracy, "f1_score": f1}
        logger.info(f"Model: {name} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_model = model
            best_f1 = f1

    logger.info(f"Best Model: {best_model} with F1 Score: {best_f1:.4f}")
    save_model(best_model)

    return {"best_model": best_model, "results": results}
