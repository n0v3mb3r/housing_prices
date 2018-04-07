import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

np.random.seed(1)

features = df.columns[1:(len(df.columns) - 1)]
target_variable = df.columns[len(df.columns) - 1]

raw_train = df.astype(object).replace(np.nan, 'None')
raw_test = test.astype(object).replace(np.nan, 'None')

train = pd.get_dummies(raw_train[features])
test = pd.get_dummies(raw_test)

missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns]

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(train, raw_train[target_variable])

predictions = pd.DataFrame(rfr.predict(test))
export = pd.concat([raw_test["Id"], predictions], axis = 1)
export_headers = ["Id", "SalePrice"]
export.columns = export_headers

export.to_csv("submission.csv", columns=export_headers, index = False)
