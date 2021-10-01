import sys 

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

import seaborn as sns
import matplotlib.pyplot as plt

from model import get_importance_selector, get_training_model


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def fit_and_predict(X_train, y_train, X_val, cross_val=False):

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
X_train.drop(['building_id', 'street', 'price', 'id_r', 'address'], axis=1, inplace=True)
X_val.drop(['building_id', 'street', 'price', 'id_r', 'address'], axis=1, inplace=True)
X_test = test_df.drop(['building_id', 'id_r', 'street', 'address'], axis=1)


if sys.argv[1] == "test":
    y_pred = fit_and_predict(X_train, y_train, X_test)

    # Write result to textfile
    result = np.column_stack((X_test.index.to_numpy(), y_pred))
    np.savetxt(r'.\\result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')

elif sys.argv[1] == "val":
    y_pred = fit_and_predict(X_train, y_train, X_val, cross_val=False)

    # Evaluate the model
    rmsle = rmsle(y_val, y_pred)
    print(f"RMSLE on validation set: {rmsle:.4f}\n")

