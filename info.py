import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

def print_nan_amount(data):
    print(data.isna().sum())


def plot_target_distribution(data):
    trans_data = pd.DataFrame(np.log1p(data["price"]), columns=["price"])
    
    sns.displot(data, x="price", label="target", kde=True)
    sns.displot(trans_data, x="price", label="log(1 + target)", kde=True)

    plt.legend()
    plt.show()


def plot_feature_distribution(data, num=9):
    root = int(np.sqrt(num))
    features = ["area_total", "area_kitchen", "area_living", "latitude", "longitude", "stories"]
    # data[features] = np.log1p(data[features])
    
    fig, ax = plt.subplots(root, root)
    for i, feature in enumerate(features):
        sns.histplot(data, x=feature, kde=True, ax=ax[i // root, i % root])

    plt.show()


def draw_correlation_map(data):
    sns.heatmap(data.corr(), xticklabels=True, yticklabels=True, cmap='RdYlGn', linewidths=.2, annot=True, fmt=".2f")
    plt.show()


def draw_house_locations(data):
    sns.scatterplot(x= data['latitude'], y=data['longitude'])
    plt.show()


apartments_train = pd.read_csv("data/apartments_train.csv")
buildings_train = pd.read_csv("data/buildings_train.csv")
data_df = apartments_train.merge(buildings_train, left_on='building_id', right_on='id', suffixes=('', '_r')).sort_values('id').set_index('id')
data_df.drop(['building_id', 'id_r', 'street', 'address'], axis=1, inplace=True)

# print_nan_amount(data_df)
# plot_target_distribution(data_df)
# draw_correlation_map(data_df)
# plot_feature_distribution(data_df)
# draw_house_locations(data_df)