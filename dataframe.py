import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

imputer = SimpleImputer(strategy='mean')
# can also do imputer = KNNImputer(n_neighbors=3, weights="uniform") for KNN!

df = pd.read_csv('/Users/christopherginting/Downloads/census+income+kdd/census/census-income.data')

print(df.head(100))

le = LabelEncoder()

# Fit and transform the 'Feature1' column

map_array=[]

def replace_with_nan(df_col, labels):
    for key in labels.keys():
        if key == ' Not in universe' or key == ' ?':
            df_col = df_col.replace(labels[key], np.NaN)
    return df_col

# define the column names:
column_names=[]
for i in range(1,43):
    column_names.append(str(i))

for column in column_names:
    df[column] = le.fit_transform(df[column])
    map_array.append(dict(zip(le.classes_, le.transform(le.classes_))))
    df[column] = replace_with_nan(df[column], map_array[int(column)-1]) # df[n] corresponds to the mapping in map_array[n-1]

imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed, columns = column_names)
print(df_imputed) # imputed values!