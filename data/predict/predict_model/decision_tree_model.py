from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from predict_model import SKLPredictModel
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Visualize the decision tree
# plt.figure(figsize=(26,8))
# plot_tree(regr, feature_names=X.columns, filled=True, rounded=True, impurity=False, fontsize=16, precision=2)
# plt.tight_layout()
# plt.show()

class DecisionTreePredictModel(SKLPredictModel):
    def __init__(self, tag):
        self.model_name = "decision_tree"
        self.saved_model_suffix_name = "pkl"
        self.model = DecisionTreeRegressor(random_state=42)
        super().__init__(tag)
    