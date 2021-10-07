import sys 

import pandas as pd
import numpy as np
import pickle

import sklearn.metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

import seaborn as sns
import matplotlib.pyplot as plt

from model import get_importance_selector, get_training_model


def rmsle(y_true, y_pred):
    return np.sqrt(sklearn.metrics.mean_squared_log_error(y_true, y_pred))


def get_preprocessor(X_train):
    categorical_columns = ["seller", "layout", "condition", "district", "material", "parking", "heating"]
    numerical_columns = X_train.drop(categorical_columns, axis=1).columns

    # Preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
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

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # ('importance_selector', get_importance_selector()),
    ])

    return pipeline



def fit_and_predict(X_train, y_train, X_val, preprocessor, cross_val=False):

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', get_training_model())
    ])

    if cross_val:
        print("Performing cross-validation ...")
        scores = cross_val_score(pipeline, X_train, y_train, scoring=make_scorer(rmsle), cv=5, n_jobs=-1)

        print("Cross-validation scores:", scores)
        print(f"RMSLE mean: {scores.mean():.4f}, RMSLE std: {scores.std():.4f}\n")
    else:
        pipeline.verbose = True

    # Preprocessing of training data, fit model
    print("Fitting model ...")
    pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    print("Predicting ...")
    preds = pipeline.predict(X_val)
    
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

    # Splitting data into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(train_df, train_df.price, train_size=0.85, random_state=0) #, stratify=np.log(train_df.price).round()
    X_train = X_train.drop(['building_id', 'street', 'price', 'id_r', 'address'], axis=1)
    X_val = X_val.drop(['building_id', 'street', 'price', 'id_r', 'address'], axis=1)
    X_test = test_df.drop(['building_id', 'id_r', 'street', 'address'], axis=1)

    
    if sys.argv[1] == "test":
        preprocessor = get_preprocessor(X_train)
        y_pred = fit_and_predict(X_train, y_train, X_test, preprocessor)

        # Write result to textfile
        result = np.column_stack((X_test.index.to_numpy(), y_pred))
        np.savetxt(r'./result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')

    elif sys.argv[1] == "val":
        preprocessor = get_preprocessor(X_train)
        y_pred = fit_and_predict(X_train, y_train, X_val, preprocessor, cross_val=True)

        # Evaluate the model
        rmsle = rmsle(y_val, y_pred)
        print(f"RMSLE on validation set: {rmsle:.4f}\n")

    elif sys.argv[1] == "automl":

        import autosklearn.regression
        from autosklearn.metrics import make_scorer, mean_squared_log_error, mean_squared_error
        
        preprocessor = get_preprocessor(X_train)
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)

        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=int(sys.argv[2]),
            per_run_time_limit=int(sys.argv[3]),
            tmp_folder='/proj/ciptmp/ny70konu/tmp_autosklearn',
            n_jobs=4,
            memory_limit=15000,
            metric=autosklearn.metrics.mean_squared_log_error,
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
        with open("/proj/ciptmp/ny70konu/automl_model.pkl", "wb") as f:
            pickle.dump(automl, f)