import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

class PredHeaders:
    latency = "Latency"
    decode_tkn = "Decode Tkn"
    prefill_tkn = "Prefill Tkn"
    decode_len = "Decode Len"
    prefill_cnt = "Prefill Cnt"

_default_output_pathname = "outputs"
MODEL_FOLDER = Path(os.getenv("PREDICT_MODEL_PATH", _default_output_pathname))
MODEL_FOLDER.mkdir(parents = True, exist_ok = True)

SAVED_SCALER_FILE = MODEL_FOLDER / "std_scaler.bin"

OUTPUT_FOLDER = Path(_default_output_pathname)
OUTPUT_FOLDER.mkdir(parents = True, exist_ok = True)
SAVED_TEST_DATA_CSV_FILE = OUTPUT_FOLDER / "test_data.csv"
TRAIN_DATA_FILE = OUTPUT_FOLDER / "data_train.pt"
VALIDATE_DATA_FILE = OUTPUT_FOLDER / "data_validate.pt"
TEST_DATA_FILE = OUTPUT_FOLDER / "data_test.pt"