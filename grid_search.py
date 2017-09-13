# -*- coding: utf8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def gen_datasets():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    return train_test_split(X, y, test_size=0.4, random_state=0)

def main():
    x_train, x_test, y_train, y_test = gen_datasets()
    parameters = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
                  {"kernel": ["linear"], "C": [1, 10, 100, 1000]}
                ]
    scores = ["precision", "recall"]
    for item in scores:
        print "hyper-parameters for: %s" % item
        clf = GridSearchCV(SVC(), parameters, cv=5, scoring="%s_macro" % item)
        clf.fit(x_train, y_train)

        print "Best params found here:"
        print clf.best_params_

        print "grid scores: "
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print("Detailed classification report:")
        y_true, y_pred = y_test, clf.predict(x_test)
        print classification_report(y_true, y_pred)



if __name__ == "__main__":
    main()
