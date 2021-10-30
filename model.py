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
from lightgbm import LGBMRegressor



def get_training_model():
    model_xgb = XGBRegressor(n_estimators=4000, max_depth=3)
    model_cat = CatBoostRegressor(iterations=1000, depth=7, learning_rate=0.15, l2_leaf_reg=0.45, silent=True, cat_features=[])
    model_lgbm = LGBMRegressor(n_estimators=1200, learning_rate=0.1, max_depth=7) # num_leaves, min_data_in_leaf

    trans_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
    trans_cat = TransformedTargetRegressor(regressor=model_cat, func=np.log1p, inverse_func=np.expm1)
    trans_lgbm = TransformedTargetRegressor(regressor=model_lgbm, func=np.log1p, inverse_func=np.expm1)

    final_model = RidgeCV()


    base_learners = [
        # ('xgb_tree', model_xgb),
        ('lgbm_tree', model_lgbm),
        ('catboost', model_cat)
    ]

    stacking_model = StackingRegressor(estimators=base_learners, n_jobs=-1, final_estimator=final_model)
    trans_stacking = TransformedTargetRegressor(regressor=stacking_model, func=np.log1p, inverse_func=np.expm1)

    return trans_stacking


def get_importance_selector():
    selection_model = XGBRegressor(n_estimators=1500, max_depth=4, n_jobs=-1, booster='gbtree')
    importance_selector = SelectFromModel(selection_model, threshold=-np.inf, max_features=20)
    return importance_selector