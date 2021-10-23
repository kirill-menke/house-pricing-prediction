import sys 

import pandas as pd
import numpy as np
import pickle

import sklearn.metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import VotingRegressor

import seaborn as sns
import matplotlib.pyplot as plt

from model import get_importance_selector, get_training_model


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
        ('oh_encoder', OneHotEncoder(sparse=True, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
    ])

    return preprocessor


def create_new_features(train_df, test_df):
    # Distance from red square
    center = np.array([37.621, 55.754]) # longitude, latitude
    train_df['center_distance'] = np.linalg.norm(train_df[['longitude', 'latitude']] - center, axis=1)
    test_df['center_distance'] = np.linalg.norm(test_df[['longitude', 'latitude']] - center, axis=1)

    # Average price per district
    # mean_price = train_df.groupby('district').mean().reset_index()[['district', 'price']]
    # train_df = train_df.merge(mean_price, how='left', on='district', suffixes=('', '_avg'))
    # test_df = test_df.merge(mean_price, how='left', on='district').rename(columns={'price' : 'price_avg'}).set_index(test_df.index)

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
        ('model', get_training_model())
    ])

    if cross_val:
        print("Performing cross-validation ...")
        # scores = cross_val_score(pipeline, X_train, y_train, scoring=make_scorer(rmsle), cv=5, n_jobs=-1)
        cv_results = cross_validate(pipeline, X_train, y_train, scoring=make_scorer(rmsle), cv=5, n_jobs=-1, return_estimator=True)
        
        scores = scores = cv_results["test_score"]
        print("Cross-validation scores:", scores)
        print(f"RMSLE mean: {scores.mean():.4f}, RMSLE std: {scores.std():.4f}\n")

        # Model takes average of all predictors
        model = VotingRegressor(list(enumerate(cv_results["estimator"])))
        model.estimators_ = cv_results["estimator"]
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
    X_train, X_val, y_train, y_val = train_test_split(train_df, train_df.price, train_size=0.9996, random_state=1, stratify=np.log(train_df.price).round())
    X_train = X_train.drop(['building_id', 'price', 'id_r'], axis=1)
    X_val = X_val.drop(['building_id', 'price', 'id_r'], axis=1)
    X_test = test_df.drop(['building_id', 'id_r'], axis=1)
    
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
              'model__regressor__n_estimators': np.arange(100, 2000, 100),  
              'model__regressor__max_depth': np.arange(3, 13, 1),
              'model__regressor__num_leaves': np.arange(2**3, 2**12, 250),
              'model__regressor__learning_rate': [0.1]
        }

        grid_catboost = {
              'model__regressor__iterations': np.arange(100, 2500, 100),  
              'model__regressor__depth': np.arange(3, 10, 1),
              # 'model__regressor__l2_leaf_reg': np.arange(0.1, 1.0, 0.05),
              'model__regressor__learning_rate': [0.15]
        }

        grid_xgboost = {
              'model__regressor__n_estimators': np.arange(100, 2500, 100),  
              'model__regressor__max_depth': np.arange(3, 10, 1),
              'model__regressor__reg_lambda': np.arange(0.1, 1.0, 0.05),
              'model__regressor__learning_rate': [0.1, 0.15, 0.2]
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






        