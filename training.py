#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import bentoml

# Import the data


def get_data():
    df = pd.read_csv('diabetes.csv', sep=',')
    return df

# clean the data


def clean_data(df):
    df.columns = map(str.lower, df.columns)

# set up the validation framework


def split_data(df):
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1)
    return df_full_train, df_test, df_train, df_val

# drop target columns and id


def drop_cols(data):
    target = 'outcome'
    columns_to_drop = [target]
    data.drop(columns_to_drop, axis=1, inplace=True)

# train the model and export to bento


def train_and_export(df_full_train, df_test):
    y_full = df_full_train.outcome.values

    drop_cols(df_full_train)  # drop outcome field
    drop_cols(df_test)

    dicts_full_train = df_full_train.to_dict(
        orient='records')  # encode/ vectorise
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full)

    best_xgb_params = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 10,

        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    # train the xgboost model
    best_xgb_model = xgb.train(
        best_xgb_params, dfulltrain, num_boost_round=175)

    bentoml.xgboost.save_model("diabetes_risk_model", best_xgb_model, custom_objects={
                               "dictVectorizer": dv})  # export to bento


if __name__ == "__main__":
    df = get_data()

    clean_data(df)

    df_full_train, df_test, df_train, df_val = split_data(df)

    train_and_export(df_full_train, df_test)
