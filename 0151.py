from math import radians

import numpy as np
import pandas as pd
import geopandas as gp

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.neighbors import BallTree, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics.pairwise import haversine_distances
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA


pd.options.mode.chained_assignment = None
seed = np.random.seed(0)
THREADS = 4

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def haversine_array(lat1, lng1, lat2 = 55.752, lng2 = 37.617):
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
    # Add nearest sub_area based on its center
    bulding_locs = np.asarray(list(zip(data_df['latitude'], data_df['longitude'])))
    sub_area_locs = np.asarray(list(zip(sub_area_centers['longitude'], sub_area_centers['latitude'])))
    closest_sub_idx = np.argmin(haversine_distances(bulding_locs, sub_area_locs), axis=1)
    data_df["sub_area_"] = sub_area_centers['sub_area'].iloc[closest_sub_idx].values

    # Mapping each building to its real sub_area
    geo_df = gp.GeoDataFrame(data_df, geometry=gp.points_from_xy(data_df.longitude, data_df.latitude))
    geo_df.crs = "EPSG:4326"
    data_df = gp.sjoin(sub_areas, geo_df, how='right', predicate='contains')
    data_df = data_df.drop(["DISTRICT", "geometry", "OKATO", "OKTMO", "OKATO_AO", "index_left"], axis=1)
    
    # Add distance from city center
    data_df['center_distance'] = haversine_array(data_df['latitude'], data_df['longitude'])

    # Add closest metro_station
    metro_locs = np.asarray(list(zip(metro_stations['latitude'], metro_stations['longitude'])))
    closest_metro_dist = np.min(haversine_distances(bulding_locs, metro_locs), axis=1)
    data_df['closest_metro'] = closest_metro_dist

    raion_features = ['preschool_education_centers_raion', 'school_education_centers_raion', 'school_education_centers_top_20_raion', 'healthcare_centers_raion',
        'university_top_20_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion', 'build_count_brick']

    school_features = ['preschool_education_centers_raion', 'school_education_centers_raion', 'school_education_centers_top_20_raion']

    # Add sub_area (raion) related features from sberbank dataset
    # data_df['preschool_education_centers_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().preschool_education_centers_raion)
    # data_df['school_education_centers_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().school_education_centers_raion)
    # data_df['school_education_centers_top_20_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().school_education_centers_top_20_raion)
    # data_df['healthcare_centers_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().healthcare_centers_raion)
    # data_df['university_top_20_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().university_top_20_raion)
    # data_df['culture_objects_top_25_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().culture_objects_top_25_raion)
    # data_df['shopping_centers_raion'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().shopping_centers_raion)
    # data_df['build_count_brick'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().build_count_brick)

    # data_df[school_features] = data_df[school_features].fillna(-1)
    # data_df[['pca1']] = PCA(n_components=1).fit_transform(data_df[school_features]).squeeze()
    # data_df.drop(school_features, axis=1, inplace=True)

    
    # Add building related features from sberbank dataset
    data_df['mean_green_area'] = data_df['sub_area'].map(sberbank_data.groupby("sub_area").mean().green_zone_part)

    # Ordinal Encoding for both sub_area variants
    data_df["sub_area"] = LabelEncoder().fit_transform(data_df['sub_area'])
    data_df["sub_area_"] = LabelEncoder().fit_transform(data_df['sub_area_'])

    # PCA transforming the two variants above
    pca = PCA(n_components=1)
    data_df["sub_area_pca"] = pca.fit_transform(data_df[["sub_area", "sub_area_"]]).squeeze()


    # Add average price in the neighborhood
    tree = BallTree(data_df[["latitude", "longitude"]])
    dist, ind = tree.query(data_df[["latitude", "longitude"]], k=300)
    mean_sqm_price = []
    mean_subarea_price = []

    for rows in ind:
        mean_sqm_price.append(np.nanmean(data_df['price'].iloc[rows] / np.log(data_df['area_total'].iloc[rows])))
        mean_subarea_price.append(np.nanmean(data_df['price'].iloc[rows]))


    data_df["mean_sqm_price"] = mean_sqm_price
    # data_df["mean_subarea_price"] = mean_subarea_price


    #cleaning area data
    # wrong_kitch_sq_index = data_df['area_kitchen'] > data_df['area_total']
    # data_df.loc[wrong_kitch_sq_index, 'area_kitchen'] = data_df.loc[wrong_kitch_sq_index, 'area_total'] * 1 / 3

    # wrong_life_sq_index = data_df['area_living'] > data_df['area_total']
    # data_df.loc[wrong_life_sq_index, 'area_living'] = data_df.loc[wrong_life_sq_index, 'area_total'] * 3 / 5

    #Add floor distance from the top of the building and the percentage of the floor
    # data_df['floor_from_top'] = data_df['stories'] - data_df['floor']
    # data_df['floor_over_stories'] = data_df['floor'] / data_df['stories']

    #examining year 
    # data_df['age_of_house_before_sale'] = np.where((2018 - data_df['constructed']>0), 2018 - data_df['constructed'], 0)
    # data_df['sale_before_build'] = ((2018 - data_df['constructed']) < 0).astype(int)

    # Add area percentage
    # data_df['area_kitchen_percentage']= data_df['area_kitchen'] / data_df['area_total']
    # data_df['area_living_percentage']= data_df['area_living'] / data_df['area_total']


    return data_df




def get_model():

    model_lgbm = LGBMRegressor(max_depth=6, n_estimators=1200, learning_rate=0.1, n_jobs=THREADS)
    model_xgb = XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.1)
    model_cat = CatBoostRegressor(iterations=1000, depth=7, learning_rate=0.1, silent=True)
    model_extra = ExtraTreesRegressor(n_estimators=1000)
    model_hist = HistGradientBoostingRegressor(max_depth=7, max_iter=500, learning_rate=0.1)

    trans_lgbm = TransformedTargetRegressor(regressor=model_lgbm, func=np.log1p, inverse_func=np.expm1)
    trans_xgb = TransformedTargetRegressor(regressor=model_xgb, func=np.log1p, inverse_func=np.expm1)
    trans_cat = TransformedTargetRegressor(regressor=model_cat, func=np.log1p, inverse_func=np.expm1)
    trans_extra = TransformedTargetRegressor(regressor=model_extra, func=np.log1p, inverse_func=np.expm1)
    trans_hist = TransformedTargetRegressor(regressor=model_hist, func=np.log1p, inverse_func=np.expm1)

    final_model = RidgeCV()

    base_learners = [
        ('xgb_tree', model_xgb),
        ("lgbm", model_lgbm),
        ('catboost', model_cat),
        ('extra_trees', model_extra)
    ]

    model_stacking = StackingRegressor(estimators=base_learners, final_estimator=final_model, cv=5, n_jobs=THREADS)
    trans_stacking = TransformedTargetRegressor(regressor=model_stacking, func=np.log1p, inverse_func=np.expm1)

    return trans_lgbm



if __name__ == "__main__":

    # Reading train/test data
    apartments_train = pd.read_csv("data/apartments_train.csv")
    buildings_train = pd.read_csv("data/buildings_train.csv")
    apartments_test = pd.read_csv("data/apartments_test.csv")
    buildings_test = pd.read_csv("data/buildings_test.csv")
    metro_stations = pd.read_csv("data/metro_stations.csv")

    # External datasources
    sub_area_centers = pd.read_csv("data/sberbank_sub_areas.csv")
    sub_areas = gp.read_file('data/mo_kag_SRHM.shp')
    sberbank_data = pd.read_csv("data/sberbank.csv")

    # Merge Tables: Apartments and Buildings
    train_df = apartments_train.merge(buildings_train, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')
    test_df = apartments_test.merge(buildings_test, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')

    # Merge train and test data
    data_df = pd.concat([train_df, test_df])

    # Preprocessing data
    preprocess(data_df)
    
    # Add new features
    data_df = add_features(data_df)

    data_df = data_df.drop(["address", "street", "windows_court", "windows_street", "elevator_service", "elevator_passenger", "garbage_chute", 
        "layout", "parking", "heating", "elevator_without", "district", "phones", "building_id", "id_r", "material"], axis = 1)

    categorical_features = ["condition", "seller", "street_and_address", "new"] # , "sub_area", "constructed"
    num_features = list(data_df.drop(categorical_features + ["price"], axis=1).columns)

    # Adding missing values as separate features
    # nan_features = data_df.drop('price', axis=1).isna().astype(int)

    # Impute missing values
    # data_df[num_features + categorical_features] = IterativeImputer().fit_transform(data_df[num_features + categorical_features])
    # data_df[num_features] = SimpleImputer(strategy="mean").fit_transform(data_df[num_features])
    # data_df[categorical_features] = SimpleImputer(strategy="most_frequent").fit_transform(data_df[categorical_features])

    # data_df = pd.merge(data_df, nan_features, left_index=True, right_index=True)
    
    data_df.fillna(-999, inplace = True)

    # oh_encoder = OneHotEncoder(sparse=False)
    # data_df = data_df.join(pd.DataFrame(oh_encoder.fit_transform(data_df[categorical_features])))

    # Splitting into train and test data
    X_train = data_df[0:len(train_df)]
    X_train = X_train.drop('price', axis=1)
    X_test = data_df[len(train_df):len(data_df)].drop('price', axis=1)
    y_train = train_df['price']


    # Perform cross-validation and prediction on test set
    errors = []
    preds = []

    cv = StratifiedKFold(shuffle=True)
    model = get_model()


    for train_idx, val_idx in cv.split(X_train, np.log(y_train).round()): # groups=train_df['building_id']

        X_train_, y_train_ = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model.fit(X_train_, y_train_)
        pred = model.predict(X_val)
        errors.append(rmsle(y_val, pred))

        preds.append(model.predict(X_test))

        print("Error: ", errors[-1])
    print("Mean error: ", np.mean(errors))


    # y_pred = np.mean(preds, axis = 0)
    y_pred = model.fit(X_train, y_train).predict(X_test)
    result = np.column_stack((X_test.index.to_numpy(), y_pred))
    np.savetxt(r'./result.csv', result, fmt=['%d', ' %.3f'], delimiter=',', header="id,price_prediction", comments='')

# Current model
# LGBM: 0.12645358031498863 (0.15112)
# 0.12526345616871512

# Added stuff which improve the score but not uploaded:
# - Changing position to Kremlin slightly: 55.75, 37.6 -> 55.752, 37.617
# - Removing log of area_total
# - Added green areas
# (- Adding average price to balltree)

# TODO
# Predict price/sqm?
# BallTree nur auf trainingsdaten
# Remove address merging

# Average sqm preis und preis pro sub_area
# Parameter tuning
# EDA auf derzeitigem Stand (Different feature importance techniques)
# Read word document
# LightGBM as final model for stacking
# Distance to financial district
# Distance from sub_area center to metro
# AutoML: MLJAR, H20 (-> Restricting AutoML: Exclude preprocessing)
# Target Encoding (e.g. LightGBM)
# Look how stacking of sklearn works
# Add HistGradientBoostingRegressor
# Visualize predictions
# Try lgbm cross validation and early stopping
# Fill missing values in full_sq, life_sq, kitch_sq, num_room by using median of apartments in the proximity (e.g. sub_area)
# Remove points with big difference between prediction and ground truth (Using simple xgboost): abs(y_pred - y_true)/y

# Questions:
# StratifiedGroupSplit not representative to leaderboard score (To which category distribution should be pay attention?)
# Were these houses actually sold for these prices or only offered? Timestamp of transaction?
# Stratified Split zu wenig Klassen
# Data Cleaning (only important features? E.g. distance to kremling < 100)
# Any Imputing other than -999 makes result worse?
# OH-Encoding makes predictions worse (At least locally)?
# Viele industrielle Regionen?
# How to handle too large sqm sizes?