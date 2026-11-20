
from config import (
    PROJECT_NAME,
    RAW_DATA_PATH,
    MODEL_DIR,
    REPORT_DIR
)

print("=" * 55)
print(PROJECT_NAME)
print("=" * 55)

print("Raw dataset exists:", RAW_DATA_PATH.exists())
print("Model directory exists:", MODEL_DIR.exists())
print("Reports directory exists:", REPORT_DIR.exists())

if RAW_DATA_PATH.exists():
    print("Dataset file:", RAW_DATA_PATH.name)
    print("Project setup completed successfully.")
else:
    print("Please download the dataset into data/raw/")

