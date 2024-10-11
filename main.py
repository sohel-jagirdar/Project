# main.py
from data_loading import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from feature_encoding import one_hot_encode
from feature_scaling import scale_features
from modeling import train_multiple_models
from evaluation import evaluate_model
from prediction import load_model, make_prediction

def main():
    # Load the data
    data = load_data()

    # Validate the data
    validate_data(data)

    # Clean the data
    cleaned_data = clean_data(data)

    # One-hot encoding (though not needed for Iris dataset)
    # Here, we don't have categorical columns to encode
    # data_encoded = one_hot_encode(cleaned_data, categorical_columns)

    # Scale the features
    X_scaled, y = scale_features(cleaned_data)

    # Train models and get the best one
    results = train_multiple_models(X_scaled, y)

    # Load the best model and evaluate it
    best_model = results["best_model"]
    evaluate_model(best_model, X_scaled, y)

    # Make predictions on new data (example using the first 5 instances)
    model = load_model()
    dummy_data = X_scaled[:5]  # Just an example to predict the first 5 instances
    predictions = make_prediction(model, dummy_data)

    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
