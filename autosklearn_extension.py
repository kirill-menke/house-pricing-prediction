from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS


class CatBoostRegressor_ext(AutoSklearnRegressionAlgorithm):

    def __init__(self, iterations, learning_rate, depth, l2_leaf_reg, random_state=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.estimator = None


    def fit(self, X, y):
        from sklearn.compose import TransformedTargetRegressor
        from catboost import CatBoostRegressor
        import numpy as np

        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)

        regressor_cat = CatBoostRegressor(
            iterations=self.iterations, 
            depth=self.depth,
            learning_rate=self.learning_rate, 
            l2_leaf_reg=self.l2_leaf_reg,
        )

        self.estimator = regressor_cat # TransformedTargetRegressor(regressor=regressor_cat, func=np.log1p, inverse_func=np.expm1)

        self.estimator.fit(X, y)

        return self

    
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        y_preds = self.estimator.predict(X)
        return y_preds

    
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
            'input': (DENSE, SIGNED_DATA, UNSIGNED_DATA),
            'output': (PREDICTIONS, UNSIGNED_DATA)
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

        cs.add_hyperparameters([iterations, learning_rate, depth, l2_leaf_reg])

        return cs
    

class KernelRidgeRegression(AutoSklearnRegressionAlgorithm):

    def __init__(self, alpha, kernel, gamma, degree, coef0, random_state=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)
        self.coef0 = float(self.coef0)

        import sklearn.kernel_ridge
        self.estimator = sklearn.kernel_ridge.KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
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
            'shortname': 'KRR',
            'name': 'Kernel Ridge Regression',
            'handles_regression': True,
            'handles_classification': False,
            'handles_multiclass': False,
            'handles_multilabel': False,
            'handles_multioutput': True,
            'is_deterministic': True,
            'input': (SPARSE, DENSE, UNSIGNED_DATA), # , SIGNED_DATA
            'output': (PREDICTIONS,)
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name='alpha', lower=10 ** -5, upper=1, log=True, default_value=1.0
        )
        kernel = CategoricalHyperparameter(
            name='kernel',
            # We restrict ourselves to two possible kernels for this example
            choices=['polynomial', 'rbf'],
            default_value='polynomial'
        )
        gamma = UniformFloatHyperparameter(
            name='gamma', lower=0.00001, upper=1, default_value=0.1, log=True
        )
        degree = UniformIntegerHyperparameter(
            name='degree', lower=2, upper=5, default_value=3
        )
        coef0 = UniformFloatHyperparameter(
            name='coef0', lower=1e-2, upper=1e2, log=True, default_value=1,
        )
        cs.add_hyperparameters([alpha, kernel, gamma, degree, coef0])
        degree_condition = EqualsCondition(degree, kernel, 'polynomial')
        coef0_condition = EqualsCondition(coef0, kernel, 'polynomial')
        cs.add_conditions([degree_condition, coef0_condition])
        return cs


# if __name__ == "__main__":
#     cs = CatBoostRegressor_ext.get_hyperparameter_search_space()
#     print(cs)