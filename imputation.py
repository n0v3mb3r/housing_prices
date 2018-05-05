# Function to impute a data frame with the median for numeric values and "_Other_" for strings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


# load data
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
variable_types = pd.read_csv("variable_types.csv", index_col = 0)

int_vars = variable_types[variable_types['type'] == 'int']['var_name']

df[int_vars].astype(object).replace(np.nan,)