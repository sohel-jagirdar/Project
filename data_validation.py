# data_validation.py
from logging_config import setup_logger

logger = setup_logger('data_validation', 'data_validation.log')


def validate_data(data):
    missing_values = data.isnull().sum()
    data_types = data.dtypes
    logger.info("Data validation completed.")

    validation_summary = {
        'missing_values': missing_values,
        'data_types': data_types
    }

    logger.info(f"Validation Summary: {validation_summary}")
    return validation_summary
