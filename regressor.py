import sys 

import pandas as pd
import numpy as np
import pickle

import sklearn.metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import BallTree

import seaborn as sns
import matplotlib.pyplot as plt

from model import get_importance_selector, get_training_model
from transformers import MeanSqmPrice


def rmsle(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_log_error(y_true, y_pred))


def get_preprocessor(X_train):
    categorical_columns = ["condition", "district", "heating", "parking", "material", "layout", "seller", 
    "windows_court", "windows_street", "new", "elevator_without", "elevator_passenger", "elevator_service", "garbage_chute", "street", "address"]
    numerical_columns = X_train.drop(categorical_columns, axis=1).columns

    # Preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('oh_encoder', OneHotEncoder(sparse=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
    ])

    return preprocessor


def create_new_features(train_df, test_df):

    # Encode string adresses to integers
    string_encoder = LabelEncoder()
    train_df["address"] = string_encoder.fit_transform(train_df["address"])
    train_df["street"] = string_encoder.fit_transform(train_df["street"])

    # Replacing wrong locations with correct
    train_df.loc[(train_df["street"] == "Бунинские Луга ЖК") & (train_df["address"] == "к2/2/1"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    train_df.loc[(train_df["street"] == "Бунинские Луга ЖК") & (train_df["address"] == "к2/2/2"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    train_df.loc[(train_df["street"] == "улица 1-я Линия") & (train_df["address"] == "57"), ["latitude", "longitude"]] = [55.6324711, 37.4536057]
    train_df.loc[(train_df["street"] == "улица Центральная") & (train_df["address"] == "75"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    train_df.loc[(train_df["street"] == "улица Центральная") & (train_df["address"] == "48"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    
    test_df.loc[(test_df["street"] == "Бунинские Луга ЖК") & (test_df["address"] == "к2/2/1"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    test_df.loc[(test_df["street"] == "Бунинские Луга ЖК") & (test_df["address"] == "к2/2/2"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    test_df.loc[(test_df["street"] == "улица 1-я Линия") & (test_df["address"] == "57"), ["latitude", "longitude"]] = [55.6324711, 37.4536057]
    test_df.loc[(test_df["street"] == "улица Центральная") & (test_df["address"] == "75"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    test_df.loc[(test_df["street"] == "улица Центральная") & (test_df["address"] == "48"), ["latitude", "longitude"]] = [55.5415152, 37.4821752]
    
    # Replacing Nan locations
    train_df.loc[(train_df["street"] == "пос. Коммунарка Москва") & (train_df["address"] == "А101 ЖК"), ["latitude", "longitude"]] = [55.5676692, 37.4816608]
    test_df.loc[(test_df["street"] == "пос. Коммунарка") & (test_df["address"] == "Москва А101 ЖК"), ["latitude", "longitude"]] = [55.5676692, 37.4816608]

    # Rescaling out of scale ceilings
    # train_df.ceiling[train_df.ceiling > 200] = train_df.ceiling/100
    # train_df.ceiling[(train_df.ceiling > 25) & (train_df.ceiling < 200)] = train_df.ceiling/10

    # Unifying features
    # train_df["street_and_address"] = train_df.street + " " + train_df.address
    # train_df["bathrooms"] = train_df.bathrooms_shared  + train_df.bathrooms_private
    # train_df["balconies_and_loggias"] = train_df.balconies + train_df.loggias

    # Dropping duplicates
    # train_df = train_df.drop(train_df[train_df.duplicated()].index, axis = 0)

    # Average sqm price in neighborhood of each house
    # tree = BallTree(train_df[["latitude", "longitude"]])

    # dist_train, ind_train = tree.query(train_df[["latitude", "longitude"]], k=3)
    # dist_test, ind_test = tree.query(test_df[["latitude", "longitude"]], k=3)

    # mean_sqm_price_of_cluster_train = []
    # for row in ind_train:
    #     mean_sqm_price_of_cluster_train.append(np.mean(train_df.price.iloc[row]/train_df.area_total.iloc[row]))

    # mean_sqm_price_of_cluster_test = []
    # for row in ind_test:
    #     mean_sqm_price_of_cluster_test.append(np.mean(train_df.price.iloc[row]/train_df.area_total.iloc[row]))

    # train_df["mean_sqm_price_of_cluster"] = mean_sqm_price_of_cluster_train
    # test_df["mean_sqm_price_of_cluster"] = mean_sqm_price_of_cluster_test

    # Distance from red square
    center = np.array([37.621, 55.754]) # longitude, latitude
    train_df['center_distance'] = np.linalg.norm(train_df[['longitude', 'latitude']] - center, axis=1)
    test_df['center_distance'] = np.linalg.norm(test_df[['longitude', 'latitude']] - center, axis=1)

    # Price per sqm district
    # price_per_district = train_df.groupby("district").mean().price
    # area_per_district = train_df.groupby("district").mean().area_total

    # train_df["price_per_sq_dist_cat"], bins = pd.qcut(train_df.center_distance, q = 150, retbins = True)
    # price_per_dist = train_df.groupby("price_per_sq_dist_cat").mean().price
    # area_per_dist = train_df.groupby("price_per_sq_dist_cat").mean().area_total

    # a = np.log(price_per_district/area_per_district)
    # b = np.log(price_per_dist/area_per_dist)

    # train_df = pd.concat([train_df, pd.DataFrame(columns = ["sqm_per_dist"])], axis = 1)
    # train_df["district"] = train_df["district"].map(a).astype(float)
    # train_df["sqm_per_dist"] = train_df["price_per_sq_dist_cat"].map(b).astype(float)
    # train_df = train_df.drop("price_per_sq_dist_cat", axis = 1)

    # Distance to nearest metro station
    metro_locs = pd.read_csv("data/metro_stations.csv").to_numpy()
    house_locs_train = train_df[['latitude','longitude']].to_numpy()
    house_locs_test = test_df[['latitude','longitude']].to_numpy()

    dists_train = np.min(np.linalg.norm(house_locs_train[:, np.newaxis, :] - metro_locs, axis=2), axis=1)
    dists_test = np.min(np.linalg.norm(house_locs_test[:, np.newaxis, :] - metro_locs, axis=2), axis=1)
    train_df['closest_metro'] = dists_train
    test_df['closest_metro'] = dists_test

    return train_df, test_df


def fit_and_predict(X_train, y_train, X_val, preprocessor, cross_val=False):

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # ("mean_sqm", MeanSqmPrice(y_train, X_train['area_total'])),
        ('model', get_training_model())
    ])

    if cross_val:
        print("Performing cross-validation ...")
        
        # Exclusion of building IDs
        cv = StratifiedGroupKFold()
        errors = []
        estimators = []
        pd.options.mode.chained_assignment = None
        for train_idxs, val_idxs in cv.split(X_train, np.log(y_train).round(), groups=X_train['building_id']):
            X_train_l, X_val_l = X_train.iloc[train_idxs], X_train.iloc[val_idxs]
            y_train_l, y_val_l = y_train.iloc[train_idxs], y_train.iloc[val_idxs]
            
            # Add Clustering: Average sqm price in the neighborhood
            # tree = BallTree(X_train_l[["latitude", "longitude"]])
            # dist_train, ind_train = tree.query(X_train_l[["latitude", "longitude"]], k=5)
            # dist_val, ind_val = tree.query(X_val_l[["latitude", "longitude"]], k=5)

            # mean_sqm_price_of_cluster_train = []
            # for row in ind_train:
            #     mean_sqm_price_of_cluster_train.append(np.mean(y_train_l.iloc[row]/X_train_l.area_total.iloc[row]))
            

            # mean_sqm_price_of_cluster_val = []
            # for row in ind_val:
            #     mean_sqm_price_of_cluster_val.append(np.mean(y_train_l.iloc[row]/X_train_l.area_total.iloc[row]))

            # X_train_l["mean_sqm_price_of_cluster"] = y_train_l # mean_sqm_price_of_cluster_train
            # X_val_l["mean_sqm_price_of_cluster"] = y_val_l # mean_sqm_price_of_cluster_val

            estimators.append(pipeline.fit(X_train_l, y_train_l))
            preds = pipeline.predict(X_val_l)

            error = rmsle(y_val_l, preds)
            errors.append(error)
            print("RMSLE:", error)

        print("Average: ", np.mean(errors))

        # cv_results = cross_validate(pipeline, X_train, y_train, scoring=make_scorer(rmsle), cv=cv, groups=X_train['building_id'], n_jobs=-1, return_estimator=True)        
        # scores = scores = cv_results["test_score"]
        # print("Cross-validation scores:", scores)
        # print(f"RMSLE mean: {scores.mean():.4f}, RMSLE std: {scores.std():.4f}\n")
        

        # Model takes average of all predictors
        model = VotingRegressor(list(enumerate(estimators)))
        model.estimators_ = estimators
        model.le_ = LabelEncoder().fit(y_train)
        model.classes_ = model.le_.classes_


    else:
        pipeline.verbose = True
        
        # Preprocessing of training data, fit model
        print("Fitting model ...")
        pipeline.fit(X_train, y_train)
        model = pipeline

    # Preprocessing of validation data, get predictions
    print("Predicting ...")
    preds = model.predict(X_val)
    
    return preds


if __name__ == "__main__":

    # Reading CSV-files
    apartments_train = pd.read_csv("data/apartments_train.csv")
    buildings_train = pd.read_csv("data/buildings_train.csv")
    apartments_test = pd.read_csv("data/apartments_test.csv")
    buildings_test = pd.read_csv("data/buildings_test.csv")

    # Merge Tables: Apartments and Buildings
    train_df = apartments_train.merge(buildings_train, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')
    test_df = apartments_test.merge(buildings_test, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')

    # Adding new features
    train_df, test_df = create_new_features(train_df, test_df)

    # Splitting data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(train_df, train_df.price, train_size=0.99965, random_state=42, stratify=np.log(train_df.price).round())
    X_train = X_train.drop(['id_r', 'price'], axis=1) # 'building_id'
    X_val = X_val.drop(['price', 'id_r'], axis=1) # 'building_id'
    X_test = test_df.drop(['id_r'], axis=1) # 'building_id'


    if sys.argv[1] == "test":
        preprocessor = get_preprocessor(X_train)
        y_pred = fit_and_predict(X_train, y_train, X_test, preprocessor, cross_val=True)

        # Write result to textfile
        result = np.column_stack((X_test.index.to_numpy(), y_pred))
        np.savetxt(r'./result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')

    elif sys.argv[1] == "val":
        preprocessor = get_preprocessor(X_train)
        y_pred = fit_and_predict(X_train, y_train, X_val, preprocessor, cross_val=True)
        
        # Evaluate the model
        rmsle = rmsle(y_val, y_pred)
        print(f"RMSLE on validation set: {rmsle:.4f}\n")

    elif sys.argv[1] == "grid_search":
        pipeline = Pipeline(steps=[
            ('preprocessor', get_preprocessor(X_train)),
            ('model', get_training_model())
        ])
        
        grid_lgbm = {
              'model__regressor__n_estimators': np.arange(3900, 5100, 100),  
              'model__regressor__max_depth': [8],
              'model__regressor__num_leaves': np.arange(2**3, 2**12, 100),
              'model__regressor__learning_rate': [0.1]
        }

        grid_catboost = {
              'model__regressor__iterations': np.arange(2400, 3500, 100),  
              'model__regressor__depth': np.arange(3, 10, 1),
              # 'model__regressor__l2_leaf_reg': np.arange(0.1, 1.0, 0.05),
              'model__regressor__learning_rate': [0.15]
        }

        grid_xgboost = {
              'model__regressor__n_estimators': np.arange(100, 2500, 100),  
              'model__regressor__max_depth': np.arange(3, 12, 1),
              # 'model__regressor__reg_lambda': np.arange(0.1, 1.0, 0.05),
              # 'model__regressor__learning_rate': [0.1, 0.15, 0.2]
        }
        
        grid = GridSearchCV(pipeline, grid_catboost, scoring=make_scorer(rmsle, greater_is_better=False), cv=5, refit=True, verbose=2, n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Best params: ", grid.best_params_)
        print("Best RMSLE:", abs(grid.best_score_))


    elif sys.argv[1] == "automl":

        import autosklearn.regression
        from autosklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error
        from autosklearn_extension import CatBoostRegressor_ext, KernelRidgeRegression

        # Preprocessing data before fitting with AutoML
        preprocessor = get_preprocessor(X_train)
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)

        # Trying log-transform
        # y_train = np.log1p(y_train)
        # y_val = np.log1p(y_val)

        # Adding CatBoost to AutoML
        #autosklearn.pipeline.components.regression.add_regressor(CatBoostRegressor_ext)
        #autosklearn.pipeline.components.regression.add_regressor(KernelRidgeRegression)

        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=int(sys.argv[2]),
            per_run_time_limit=int(sys.argv[3]),
            tmp_folder='/proj/ciptmp/ny70konu/tmp_autosklearn',
            n_jobs=4,
            memory_limit=20000,
            metric=autosklearn.metrics.mean_squared_log_error,
            include = {
                'regressor': ['gradient_boosting'],
                # 'feature_preprocessor': ["kernel_pca"]
            },
            delete_tmp_folder_after_terminate=False,
            # resampling_strategy='cv',
            # resampling_strategy_arguments={'folds': 5},
            scoring_functions=[autosklearn.metrics.mean_squared_log_error]
        )

        automl.fit(X_train, y_train, X_val, y_val, dataset_name='moscow-house-pricing')

        # Final constructed ensemble 
        print(automl.show_models())
        # Statistics about run
        print(automl.sprint_statistics())
        # Models found
        print(automl.leaderboard())

        poT = automl.performance_over_time_
        poT.plot(
            x='Timestamp',
            kind='line',
            legend=True,
            title='Auto-sklearn accuracy over time',
            grid=True,
        )
        plt.savefig("Performance over time")

        # Refitting found ensemble with whole data
        # automl.refit(X_train, y_train)

        # Evaluate the model on validation data
        y_pred = automl.predict(X_val)
        rmsle = rmsle(y_val, y_pred)
        print(f"RMSLE on validation set: {rmsle:.4f}\n")

        # Evaluate the model on test data
        X_test_idx = X_test.index.to_numpy()
        X_test = preprocessor.transform(X_test)
        y_pred = automl.predict(X_test)
        result = np.column_stack((X_test_idx, y_pred))
        np.savetxt(r'./result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')

        # Save model
        with open("/proj/ciptmp/ny70konu/automl_model_gb.pkl", "wb") as f:
            pickle.dump(automl, f)


    elif sys.argv[1] == "load":
        model = pickle.load(open("/proj/ciptmp/ny70konu/automl_model_gb.pkl", 'rb'))
        
        # Preprocessing data before refiting with AutoML
        preprocessor = get_preprocessor(X_train)
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)

        # Refiting
        model.refit(X_train, y_train)

        # Evaluate the model on test data
        X_test_idx = X_test.index.to_numpy()
        X_test = preprocessor.transform(X_test)
        y_pred = automl.predict(X_test)
        result = np.column_stack((X_test_idx, y_pred))
        np.savetxt(r'./result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')






        