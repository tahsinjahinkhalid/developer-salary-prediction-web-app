from pprint import pprint
import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# model evaluation stuff
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    allow_random_CV = False
    allow_gridCV = True
    data_salaries = pd.read_pickle("data/data_predict.pkl")
    # print(data_salaries)
    features = data_salaries.drop("Salary", axis=1)
    target = data_salaries["Salary"]

    rf_mdl = RandomForestRegressor()

    if allow_random_CV:
        param_grid_random = {'bootstrap': [True, False],
                             'max_depth': [10, 20, 50,
                                           100, None],
                             'max_features': ['auto', 'sqrt'],
                             'min_samples_leaf': [1, 2, 4],
                             'min_samples_split': [2, 5, 10],
                             'n_estimators': [100, 200,
                                              500]}

        rf_random = RandomizedSearchCV(estimator=rf_mdl,
                                       param_distributions=param_grid_random,
                                       n_iter=100, cv=3,
                                       verbose=10,
                                       random_state=3007,
                                       n_jobs=-1)
        # Fit the random search model
        rf_random.fit(features, target.values)

        pprint(rf_random.best_params_)

    # param grid made based on random search CV
    if allow_gridCV:
        param_grid_CV = {'bootstrap': [True],
                         'max_depth': [10, 15, 20, 25, 30],
                         'max_features': ['sqrt'],
                         'min_samples_leaf': [2, 4, 6],
                         'min_samples_split': [2, 3, 5, 10],
                         'n_estimators': [500]}

        grid_search = GridSearchCV(
            estimator=rf_mdl,
            param_grid=param_grid_CV,
            cv=3,
            n_jobs=-1,
            verbose=10)

        # initiate grid search
        grid_search.fit(features, target.values)

        # show the best params
        pprint(grid_search.best_params_)
