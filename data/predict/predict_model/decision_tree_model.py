from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from predict_model import PredictModel
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Visualize the decision tree
# plt.figure(figsize=(26,8))
# plot_tree(regr, feature_names=X.columns, filled=True, rounded=True, impurity=False, fontsize=16, precision=2)
# plt.tight_layout()
# plt.show()

class DecisionTreePredictModel(PredictModel):
    def __init__(self, tag):
        self.model_name = "decision_tree"
        self.saved_model_suffix_name = "pkl"
        self.get_saved_model_name(tag)
        self.model = DecisionTreeRegressor(random_state=42)
    
    def train(self, X_data, y_data, train_config = {}):
        self.model.fit(X_data, y_data)
        _, test_loss = self.eval(X_data, y_data)
        print(f"📉 Final Train Loss (MSE): {test_loss:.4f}")

    def eval(self, X_data, y_data = None):
        test_outputs = self.model.predict(X_data)
        mse = None
        if isinstance(y_data, np.ndarray):
            mse = mean_squared_error(test_outputs, y_data)
        # r2 = r2_score(test_outputs, y_data)
        return test_outputs, mse

    def save(self):
        with open(self.save_model_name, "wb") as model_file:
            pickle.dump(self.model, model_file)
    
    def load(self):
        if os.path.exists(self.save_model_name):
            with open(self.save_model_name, "rb") as model_file:
                self.model = pickle.load(model_file)
