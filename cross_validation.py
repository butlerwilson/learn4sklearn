# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

def main():
    iris = datasets.load_iris()
    print iris.data.shape, iris.target.shape
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
    clf = svm.SVC(kernel="linear").fit(X_train, y_train)
    print clf.score(X_test, y_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print scores
    print scores.mean(), scores.std()

if __name__ == '__main__':
    main()