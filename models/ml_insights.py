import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, explained_variance_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def normalize(x):
    abs_sum = np.sum(np.abs(x))
    return np.round(x/abs_sum, 3)


class XGBoostShapInsight:
    def __init__(self) -> None:
        self.model = xgb.XGBRegressor(eval_metric=explained_variance_score)
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()
        self.explainer = None

    def fit(self, X:np.ndarray, 
            y:np.ndarray, 
            standardize=False,
            valid_threshold=0.8):
        if standardize:
            X = self.scaler_x.fit_transform(X)
            y = self.scaler_y.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        self.model.fit(X_train,y_train, eval_set=[(X_test, y_test)])
        evs = self.model.evals_result_['validation_0']['explained_variance_score'][-1]
        # The model is not suitable to analysis this case when evs smaller than threshold
        assert evs >= valid_threshold, "The model analyzing tool is not trustworthy for this case"
        self.explainer = shap.Explainer(self.model)

    def analysis(self, X, y, curr_idx, base_idx):
        self.shap_values = self.explainer(X)
        shap_res = self.shap_values[curr_idx].values - self.shap_values[base_idx].values
        shap_normed_res = normalize(shap_res)
        return shap_normed_res

if __name__ == "__main__":
    """
    For a specific case, the model can be trained once and load-reuse many times until its data needs to be update.
    """
    seed = 10089
    np.random.seed(seed)
    xgb_tool = XGBoostShapInsight()
    x = np.random.random((520, 5))
    y = x[:, 0] - 2*x[:,3] + 0.5*x[:,4]
    xgb_tool.fit(x,y)
    curr_idx = 50
    base_idx = 30
    out = xgb_tool.analysis(x, y, curr_idx = curr_idx, base_idx=base_idx)
    print(out)

