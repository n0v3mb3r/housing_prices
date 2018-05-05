import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# load data
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
variable_types = pd.read_csv("variable_types.csv", index_col = 0)

# set random seed
np.random.seed(1)

# create features and target variable
features = df.columns[1:(len(df.columns) - 1)]
target_variable = df.columns[len(df.columns) - 1]

# alter test and train data

for var in features:
	if variable_types.loc[var,:] == 'str':


raw_train = df.astype(object).replace(np.nan, 'None')
raw_test = test.astype(object).replace(np.nan, 'None')

# keep altering test and train data
train = pd.get_dummies(raw_train[features])
test = pd.get_dummies(raw_test)

#fix test dataset
missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns]

# fit model
print('running rfr')
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(train, raw_train[target_variable])

print('running gbc')
gbc = GradientBoostingRegressor(n_estimators=100)
gbc.fit(train, raw_train[target_variable])

print('running svr')
svr = SVR(kernel='linear')
svr.fit(train, raw_train[target_variable])

# predict model
print('predictions')
rfr_predictions = pd.DataFrame(rfr.predict(test))
gbc_predictions = pd.DataFrame(gbc.predict(test))

# svr predictions increased error, removing
# svr_predictions = pd.DataFrame(svr.predict(test))

predictions = pd.concat([rfr_predictions, gbc_predictions], axis = 1)
predictions['avg'] = predictions.mean(axis = 1)

export = pd.concat([raw_test["Id"], predictions['avg']], axis = 1)
export_headers = ["Id", "SalePrice"]
export.columns = export_headers

# export data
export.to_csv("submission.csv", columns=export_headers, index = False)
