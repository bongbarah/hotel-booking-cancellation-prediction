import json
import pandas as pd
from config import RAW_FILE,CLEAN_FILE,RESULTS
from src.cleaning import clean_data
from src.eda import create_eda
from src.train import train_models

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Put hotel_bookings.csv here: {RAW_FILE}")
    raw=pd.read_csv(RAW_FILE)
    clean,audit=clean_data(raw)
    clean.to_csv(CLEAN_FILE,index=False)
    (RESULTS/"cleaning_audit.json").write_text(json.dumps(audit,indent=2))
    print("Cleaning audit:",audit)
    create_eda(clean)
    print("\nModel comparison:")
    print(train_models(clean).to_string(index=False))
    print("\nPipeline complete.")

if __name__=="__main__": main()
