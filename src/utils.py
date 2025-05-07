import os
import pandas as pd
import joblib

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_model(model, file_path):
    """Save a trained model to disk."""
    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(file_path):
    """Load a saved model from disk."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None