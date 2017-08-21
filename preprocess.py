#-*- coding:utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import numpy as np

def gen_datasets():
    datasets = np.array(
        [[1, 2, -2, 0],
        [-1, 2, 0, 1],
        [1, 0, 2, -2],
        [2, 0, -2, -1],
        [0, -1, 2, 0]], float)

    return datasets

def main():
    datasets = gen_datasets()
    print "origin data:"
    print datasets

    #0均值，单位方差
    standard_scaler = StandardScaler()
    scaler_datasets = standard_scaler.fit_transform(datasets)
    print scaler_datasets
    print "-" * 80

    min_max_scaler = MinMaxScaler()
    scaler_datasets = min_max_scaler.fit_transform(datasets)
    print scaler_datasets
    print "-" * 80

    max_abs_scaler = MaxAbsScaler()
    scaler_datasets = max_abs_scaler.fit_transform(datasets)
    print scaler_datasets
    print "-" * 80

    normalize = Normalizer(norm="l1")
    normalize_datasets = normalize.fit_transform(datasets)
    print normalize_datasets
    print "-" * 80

    binarizer = Binarizer(threshold=1.1)
    binarizer_datasets = binarizer.fit_transform(datasets)
    print binarizer_datasets
    print "-" * 80

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder_datasets = one_hot_encoder.fit_transform([[0, 1, 4], [1, 2, 0], [2, 3, 5]])
    print one_hot_encoder_datasets.toarray()
    print "-" * 80

    imputer = Imputer(missing_values=0, strategy="median")
    imputer_datasets = imputer.fit_transform(datasets)
    print imputer_datasets
    print imputer.statistics_

if __name__ == "__main__":
    main()