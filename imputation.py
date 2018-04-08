# Function to impute a data frame with the median for numeric values and "_Other_" for strings



def impute_df(df):

    for variable_name in df.columns:
        print(df[variable_name].dtypes)

    return df