from catboost import CatBoostRegressor

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

from autosklearn.pipeline.constants import SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

import autosklearn

class CatBoostRegressor_ext():

    def __init__(self, iterations, learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature):
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.random_strength = float(random_strength)
        self.bagging_temperature = float(bagging_temperature)
        self.estimator = None


    def fit(self, X, y):
        self.estimator = CatBoostRegressor(
            iterations=self.iterations, 
            learning_rate=self.learning_rate, 
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, 
            random_strength=self.random_strength, 
            bagging_temperature=self.bagging_temperature
        )

        self.estimator.fit(X, y)

        return self

    
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError

        return self.estimator.predict(X)

    
    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'CatBoost',
            'name': 'CatBoost Regressor',
            'handles_regression': True,
            'handles_classification': False,
            'handles_multiclass': False,
            'handles_multilabel': False,
            'handles_multioutput': False,
            'is_deterministic': True,
            'input': (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            'output': (PREDICTIONS,)
        }


    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        iterations = UniformIntegerHyperparameter(
            name='iterations', lower=100, upper=3000, default_value=1000
        )

        learning_rate = UniformFloatHyperparameter(
            name='learning_rate', lower=0.001, upper=3, default_value=0.1, log=True
        )

        depth = UniformIntegerHyperparameter(
            name='depth', lower=2, upper=12, default_value=4
        )

        l2_leaf_reg = UniformFloatHyperparameter(
            name='l2_leaf_reg', lower=0.01, upper=1.0, log=True
        )

        random_strength = UniformFloatHyperparameter(
            name='random_strength', lower=0.01, upper=100.0, log=True
        )

        bagging_temperature = UniformFloatHyperparameter(
            name='bagging_temperature', lower=0.01, upper=100.0, log=True
        )

        cs.add_hyperparameters([iterations, learning_rate, depth, l2_leaf_reg])

        return cs
    

# Add KRR component to auto-sklearn.
autosklearn.pipeline.components.regression.add_regressor(KernelRidgeRegression)
cs = KernelRidgeRegression.get_hyperparameter_search_space()
print(cs)