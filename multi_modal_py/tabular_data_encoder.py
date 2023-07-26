from sklearn.preprocessing import LabelEncoder
import numpy as np


def catergorical_function(examples, Cat_feats):
    encoder = LabelEncoder()
    for col in Cat_feats:
        examples[col] = encoder.fit_transform(examples[col])
        examples[col] = examples[col].astype(np.int64)
    return examples

def catergorical_function_new(train,test, Cat_feats):
    encoder = LabelEncoder()
    for col in Cat_feats:
        train[col] = encoder.fit_transform(train[col])
        train[col] = train[col].astype(np.int64)
        test[col] = encoder.transform(test[col])
        test[col] = test[col].astype(np.int64)
    return train, test



def auto_numpy_encode(df_data, Categorical_features, Numerical_features):
    df_numpy_data = {}
    for feature in Categorical_features:
        df_numpy_data[feature] = LabelEncoder().fit_transform(df_data[feature])
    for feature in Numerical_features:
        df_numpy_data[feature] = np.array(df_data[feature])
    data = np.column_stack(df_numpy_data.values())
    label = np.array(df_data['labels'])
    return data, label