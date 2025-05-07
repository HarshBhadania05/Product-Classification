from src import config
from src.utils import load_csv
from src.eda import run_eda
from src.preprocessing import preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_models

import sys
import contextlib

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

def main():
    with open(config.OUTPUT_LOG_PATH, 'w') as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = sys.stderr = Tee(sys.stdout, f)

        try:
            # Load data
            df = load_csv(config.DATA_PATH)
            if df is None:
                print("Failed to load data.")
                return

            # Run EDA
            run_eda(df)

            # Preprocess data
            X_train, X_test, y_train, y_test, vectorizer, le = preprocess_data(df)

            # Train models
            models = train_models(X_train, y_train)

            # Evaluate models
            evaluate_models(models, X_test, y_test, le)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()
