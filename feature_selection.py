#-*- coding: utf-8 -*-

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer

from datasets import Datasets

def gen_datasets(raw_datasets):
    X = []
    Y = []
    for i in xrange(0, len(raw_datasets.data), 100):
        X.append(raw_datasets.data[i])
        Y.append(raw_datasets.target[i])

    return np.array(X), np.array(Y)

def main():
    raw_datasets,_ = Datasets.load_datasets()
    X, Y = gen_datasets(raw_datasets)

    vectorizer = CountVectorizer(decode_error="ignore")
    cv_datasets = vectorizer.fit_transform(X).toarray()

    clf = ExtraTreesClassifier()
    clf = clf.fit(cv_datasets, Y)
    print cv_datasets.shape

    print clf.feature_importances_

    modle = SelectFromModel(clf, prefit=True)
    X_new = modle.transform(cv_datasets)
    print X_new.shape


if __name__ == '__main__':
    main()