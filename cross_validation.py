# -*- coding: utf-8 -*-

from sklearn import datasets

from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut


def main():
    iris = datasets.load_iris()
    print iris.data.shape, iris.target.shape
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
    clf = svm.SVC(kernel="linear").fit(X_train, y_train)
    print clf.score(X_test, y_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print scores
    print scores.mean(), scores.std()

    cv = ShuffleSplit(n_splits=3, test_size=0.3)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print scores

    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, X_train, y_train, scoring=scoring)
    print scores.keys()
    print scores["test_precision_macro"]

    kf = KFold(n_splits=10)
    #for train, test in kf.split(X_train):
    #    print "%s, %s" % (train, test)

    loo = LeaveOneOut()
    #for train, test in loo.split(X_train):
    #    print "%s, %s" % (train, test)

if __name__ == '__main__':
    main()