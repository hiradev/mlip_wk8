import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    # np.sqrt works across scikit-learn versions (squared=False was removed in 1.6)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    return {"RMSE": rmse, "R2": r2}
