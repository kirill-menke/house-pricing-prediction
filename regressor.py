import sys 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

import seaborn as sns
import matplotlib.pyplot as plt

from model import get_importance_selector, get_training_model


def fit_and_predict(X_train, y_train, X_val):
    # Preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            # ('normalize', QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0), 
            # ["area_total", "area_kitchen", "area_living", "floor", "ceiling", "latitude", "longitude", "constructed", "stories"]),
            ('num', numerical_transformer, X.columns),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('importance_selector', get_importance_selector()),
        # ('pca', PCA()),
        ('model', get_training_model())
    ], verbose=True)


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
X = train_df.drop(['building_id', 'id_r', 'price', 'street', 'address'], axis=1)
y = train_df['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=0)
X_test = test_df.drop(['building_id', 'id_r', 'street', 'address'], axis=1)


if sys.argv[1] == "test":
    preds = fit_and_predict(X_train, y_train, X_test)

    # Write result to textfile
    result = np.column_stack((X_test.index.to_numpy(), preds))
    np.savetxt(r'.\\result.csv', result, fmt=['%d', '%.1f'], delimiter=',', header="id,price_prediction", comments='')

elif sys.argv[1] == "val":
    preds = fit_and_predict(X_train, y_train, X_val)

    # Evaluate the model
    rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
    print("RMSLE:", rmsle)

