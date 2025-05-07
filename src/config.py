import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "pricerunner_aggregate.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

RANDOM_STATE = 42
TEST_SIZE = 0.2

OUTPUT_LOG_PATH = os.path.join(REPORT_DIR, "output_report.txt")