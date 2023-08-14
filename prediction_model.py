import pandas as pd
# import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model evaluation stuff
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

if __name__ == "__main__":
    data_salaries = pd.read_pickle("data/data_predict.pkl")
    # print(data_salaries)
    features = data_salaries.drop("Salary", axis=1)
    target = data_salaries["Salary"]

    lin_reg_mdl = LinearRegression()
    lin_reg_mdl.fit(features, target.values)

    # predict y
    predicted = lin_reg_mdl.predict(features)

    print("=========================")
    print("Linear Regression Metrics")
    print("=========================")
    print(f"RMSE: {mean_squared_error(target.values, predicted, squared=False)}")
    print(f"MAE : {mean_absolute_error(target.values, predicted)}")
    print("=========================")

    # target_plt = px.line(x=features.index, y=target)
    # target_plt.update_layout()
    # target_plt.show()
    # print(features.index)

    dt_mdl = DecisionTreeRegressor(random_state=3007)
    dt_mdl.fit(features, target.values)
    predicted_dt = dt_mdl.predict(features)
    print("=========================")
    print("Decision Tree Metrics")
    print("=========================")
    print(f"RMSE: {mean_squared_error(target.values, predicted_dt, squared=False)}")
    print(f"MAE : {mean_absolute_error(target.values, predicted_dt)}")
    print("=========================")
    # print(f"{dt_mdl.get_params()}")

    # from grid search CV
    params_rf = {'bootstrap': True,
                 'max_depth': 15,
                 'max_features': 'sqrt',
                 'min_samples_leaf': 2,
                 'min_samples_split': 10,
                 'n_estimators': 500,
                 'random_state': 3007}

    rf_mdl = RandomForestRegressor(**params_rf)
    rf_mdl.fit(features, target.values)
    predicted_rf = rf_mdl.predict(features)
    print("=========================")
    print("Random Forest Metrics")
    print("=========================")
    print(f"RMSE: {mean_squared_error(target.values, predicted_rf, squared=False)}")
    print(f"MAE : {mean_absolute_error(target.values, predicted_rf)}")
    print("=========================")

    # output as pickle
    pickle.dump(dt_mdl, open('pickles/model_dt.pkl', 'wb'))