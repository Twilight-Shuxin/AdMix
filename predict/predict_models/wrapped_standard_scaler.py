
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from predict_models.constants import SAVED_SCALER_FILE

class WrappedStandardScaler():
    def __init__(self):
        self.init = False
    
    def transform(self, x):
        if(not self.init):
            self.init = True
            self.sc = StandardScaler()
            return self.sc.fit_transform(x)
        else:
            self.load()
            return self.sc.transform(x)

    def save(self):
        dump(self.sc, SAVED_SCALER_FILE, compress=True)
    
    def load(self):
        self.sc = load(SAVED_SCALER_FILE)
        self.init = True