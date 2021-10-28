import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics.pairwise import haversine_distances

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

pd.options.mode.chained_assignment = None
seed = np.random.seed(0)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def haversine_array(lat1, lng1, lat2 = 55.75, lng2 = 37.6):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def preprocess(data_df):
    # Merge features
    data_df["street_and_address"] = data_df.street + " " + data_df.address
    data_df["bathrooms"] = data_df.bathrooms_shared  + data_df.bathrooms_private
    data_df["balconies_and_loggias"] = data_df.balconies + data_df.loggias

    # Imputing coordinates outside of moscow and NaNs
    data_df.latitude[data_df.street_and_address == "Бунинские Луга ЖК к2/2/1"] = 55.5415152
    data_df.longitude[data_df.street_and_address == "Бунинские Луга ЖК к2/2/1"] = 37.4821752
    data_df.latitude[data_df.street_and_address == "Бунинские Луга ЖК к2/2/2"] = 55.5415152
    data_df.longitude[data_df.street_and_address == "Бунинские Луга ЖК к2/2/2"] = 37.4821752
    data_df.latitude[data_df.street_and_address == "улица 1-я Линия 57"] = 55.6324711
    data_df.longitude[data_df.street_and_address == "улица 1-я Линия 57"] = 37.4536057
    data_df.latitude[data_df.street_and_address == "улица Центральная 75"] = 55.5415152
    data_df.longitude[data_df.street_and_address == "улица Центральная 75"] = 37.4821752
    data_df.latitude[data_df.street_and_address == "улица Центральная 48"] = 55.5415152
    data_df.longitude[data_df.street_and_address == "улица Центральная 48"] = 37.4821752

    # NaNs
    data_df.latitude[data_df.street_and_address == "пос. Коммунарка Москва А101 ЖК"] = 55.5676692
    data_df.longitude[data_df.street_and_address == "пос. Коммунарка Москва А101 ЖК"] = 37.4816608

    # Encode strings to integers
    data_df["street_and_address"] = LabelEncoder().fit_transform(data_df.street_and_address)


def add_features(data_df):
    # Add nearest sub_area to each building
    bulding_locs = np.asarray(list(zip(data_df['latitude'], data_df['longitude'])))
    sub_area_locs = np.asarray(list(zip(subareas['longitude'], subareas['latitude'])))
    closest_sub_idx = np.argmin(haversine_distances(bulding_locs, sub_area_locs), axis=1)
    data_df["sub_area"] = subareas['sub_area'].iloc[closest_sub_idx].values
    data_df["sub_area"] = LabelEncoder().fit_transform(data_df.sub_area)

    # Add distance from city center
    data_df['center_distance'] = haversine_array(data_df['latitude'], data_df['longitude'])

    # Add average price in the neighborhood
    tree = BallTree(data_df[["latitude", "longitude"]])
    dist, ind = tree.query(data_df[["latitude", "longitude"]], k=300)

    # Take log of area
    data_df['area_total'] = np.log(data_df['area_total'])
    mean_sqm_price_of_cluster = []
    for row in ind:
        mean_sqm_price_of_cluster.append(np.nanmean(data_df.price.iloc[row]/data_df.area_total.iloc[row]))

    data_df["mean_sqm_price_of_cluster"] = mean_sqm_price_of_cluster


def get_model():

    model_lgbm = LGBMRegressor(max_depth=6, n_jobs=-2, n_estimators=1200, learning_rate=0.1)
    model_xgb = XGBRegressor(n_estimators=1500, max_depth=6, n_jobs=-2, learning_rate = 0.1)
    model_cat = CatBoostRegressor(iterations=1000, depth = 7, learning_rate=0.1, silent=True)

    trans_lgbm = TransformedTargetRegressor(regressor=model_lgbm, func=np.log1p, inverse_func=np.expm1)
    trans_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
    trans_cat = TransformedTargetRegressor(regressor=model_cat, func=np.log1p, inverse_func=np.expm1)

    final_model = RidgeCV()

    base_learners = [
        ('xgb_tree', model_xgb),
        ("lgbm", model_lgbm),
        ('catboost', model_cat),
    ]

    model_stacking = StackingRegressor(estimators=base_learners, n_jobs=-2, final_estimator=final_model, cv = 5)
    trans_stacking = TransformedTargetRegressor(regressor=model_stacking, func=np.log1p, inverse_func=np.expm1)

    return trans_lgbm



if __name__ == "__main__":

    # Reading CSV-files
    apartments_train = pd.read_csv("data/apartments_train.csv")
    buildings_train = pd.read_csv("data/buildings_train.csv")
    apartments_test = pd.read_csv("data/apartments_test.csv")
    buildings_test = pd.read_csv("data/buildings_test.csv")
    subareas = pd.read_csv("data/sberbank_sub_areas.csv")

    # Merge Tables: Apartments and Buildings
    train_df = apartments_train.merge(buildings_train, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')
    test_df = apartments_test.merge(buildings_test, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')


    # Merge train and test data
    data_df = pd.concat([train_df, test_df])

    # Preprocessing data
    preprocess(data_df)
    
    # Add new features
    add_features(data_df)

    data_df = data_df.drop(["address", "street", "bathrooms_shared", "bathrooms_private", "balconies", "loggias",
        "windows_court", "windows_street", "elevator_service", "elevator_passenger", "garbage_chute", 
        "layout", "parking", "heating", "elevator_without", "material", "phones", "seller", "district", "building_id", "id_r"], axis = 1)


    # Impute missing values
    data_df.fillna(-999, inplace = True)

    # Splitting into train and test data
    X_train = data_df[0:len(train_df)]
    X_train = X_train.drop('price', axis=1)
    X_test = data_df[len(train_df):len(data_df)].drop('price', axis=1)
    y_train = train_df['price']

    # Perform cross-validation and prediction on test set
    errors = []
    preds = []

    cv = StratifiedGroupKFold()
    model = get_model()

    for train_idx, test_idx in cv.split(X_train, np.log(y_train).round(), groups=train_df['building_id']):

        X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[test_idx], y_train.iloc[test_idx]

        model.fit(X_train_, y_train_)
        pred = model.predict(X_val)
        errors.append(rmsle(y_val, pred))

        preds.append(model.predict(X_test))

        print("Error: ", errors[-1])
    print("Mean error: ", np.mean(errors))


y_pred = np.mean(preds, axis = 0)
result = np.column_stack((X_test.index.to_numpy(), y_pred))
np.savetxt(r'./result.csv', result, fmt=['%d', '%.3f'], delimiter=',', header="id,price_prediction", comments='')


# Stratified Group Split
# Stacked: 0.1888743165146102 (0.15394), 0.18272110079641235 (0.15670), 0.16983476356751734 (0.3)
# LGBM: 0.19207617127201926 (k=300), 0.18963043421515094 (k=150), 0.17665640688920609 (k=50)