import numpy as np
import pandas as pd
import imputation as im
from sklearn.ensemble import RandomForestRegressor


class MLModel:

    def __init__(self):
    train = pd.DataFrame
    test = pd.DataFrame
    features = []
    target_variable = []

    def load_train_data(self, filename):
        train = pd.read_csv(filename)

    def load_test_data(self, filename):
        test = pd.read_csv(filename)

    def get_features(self):
        features = self.train.columns[1:(len(self.train.columns) - 1)]

    def get_target_variable(self):
        target_variable = self.train.columns[len(self.train.columns) - 1]
        target_variable = self.train.columns[len(self.train.columns) - 1]

housing_prices = MLModel()

housing_prices.load_train_data("train.csv")
housing_prices.load_test_data("test.csv")

housing_prices.get_features()
housing_prices.get_target_variable()

# load data
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# set random seed
np.random.seed(1)

# create features and target variable
features = df.columns[1:(len(df.columns) - 1)]
target_variable = df.columns[len(df.columns) - 1]

# alter test and train data
raw_train = df.astype(object).replace(np.nan, 'None')
raw_test = test.astype(object).replace(np.nan, 'None')

<<<<<<< HEAD
# keep altering test and train data
=======
#im.impute_df(raw_train)
print(raw_train.head(5))

>>>>>>> b74c1cc3b203edefc036db51b5d1e8dccb37a31c
train = pd.get_dummies(raw_train[features])
test = pd.get_dummies(raw_test)

#fix test dataset
missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns]

# fit model
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(train, raw_train[target_variable])

# predict model
predictions = pd.DataFrame(rfr.predict(test))
export = pd.concat([raw_test["Id"], predictions], axis=1)
export_headers = ["Id", "SalePrice"]
export.columns = export_headers

<<<<<<< HEAD
# export data
export.to_csv("submission.csv", columns=export_headers, index = False)
=======
export.to_csv("submission.csv", columns=export_headers, index=False)
>>>>>>> b74c1cc3b203edefc036db51b5d1e8dccb37a31c
