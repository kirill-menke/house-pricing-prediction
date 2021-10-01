import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def get_training_model():
    model_xgb = XGBRegressor(n_estimators=1500, max_depth=4, n_jobs=-1, booster='gbtree')
    trans_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)

    model_cat = CatBoostRegressor(iterations=2000,  depth=7, learning_rate=0.15, l2_leaf_reg=0.45, silent=True)
    trans_cat = TransformedTargetRegressor(regressor=model_cat, func=np.log1p, inverse_func=np.expm1)

    model_knn_256 = KNeighborsRegressor(n_neighbors=256)
    model_knn_512 = KNeighborsRegressor(n_neighbors=512)
    model_knn_1024 = KNeighborsRegressor(n_neighbors=1024, p=1, n_jobs=-1)

    final_model = RidgeCV()


    base_learners = [
        ('xgb_tree', trans_xgb),
        # ('knn_256', model3),
        # ('knn_512', model4),
        # ('knn_1024', model5),
        ('catboost', trans_cat)
    ]

    stacking_model = StackingRegressor(estimators=base_learners, n_jobs=-1, final_estimator=final_model)
    trans_stacking = TransformedTargetRegressor(regressor=stacking_model, func=np.log1p, inverse_func=np.expm1)

    return trans_cat


def get_importance_selector():
    selection_model = XGBRegressor(n_estimators=1500, max_depth=4, n_jobs=-1, booster='gbtree')
    importance_selector = SelectFromModel(selection_model, threshold=-np.inf, max_features=20) # 29
    return importance_selector