from sklearn.tree import DecisionTreeRegressor
from predict_models.predict_model import SKLPredictModel

# Visualize the decision tree
# plt.figure(figsize=(26,8))
# plot_tree(regr, feature_names=X.columns, filled=True, rounded=True, impurity=False, fontsize=16, precision=2)
# plt.tight_layout()
# plt.show()

class DecisionTreePredictModel(SKLPredictModel):
    def __init__(self):
        self.model_name = "decision_tree"
        self.model = DecisionTreeRegressor(random_state=42)
        super().__init__()
    