
import os
import tempfile
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from predict_models.constants import SAVED_SCALER_FILE

def atomic_joblib_dump(obj, path, compress = True):
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
        tmp_path = tmp_file.name
        dump(obj, tmp_path, compress = compress)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            print("Error happened, cleared tmp scaler file", tmp_path)
            os.remove(tmp_path)

class WrappedStandardScaler():
    def __init__(self):
        self.init = False
        self.save_path = SAVED_SCALER_FILE
        self.loaded = False
    
    def transform(self, x):
        if(not self.init):
            self.init = True
            self.sc = StandardScaler()
            return self.sc.fit_transform(x)
        else:
            self.load()
            return self.sc.transform(x)

    def save(self):
        if(not self.loaded):
            atomic_joblib_dump(self.sc, self.save_path)
            print("Saving scaler to", self.save_path)
    
    def load(self):
        if os.path.exists(self.save_path):
            self.sc = load(self.save_path)
            self.init = True
            self.loaded = True