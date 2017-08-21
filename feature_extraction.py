#-*- coding: utf-8 -*-

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from datasets import Datasets
from pprint import pprint

"""
    特征提取和特征选择不同，前者主要将任意的数据（文本，图像）转化为机器学习能够使用的数值的过程；
    而后者主要是说在这些特征之上使用的机器学习技术方案。
"""

def main():
    datasets = [
        {"city": "beijing", "age": 500, "temperature": 26},
        {"city": "shanghai", "age": 550, "temperature": 27},
        {"city": "shenzheng", "age": 300, "temperature": 30},
    ]

    dict_vectorizer = DictVectorizer()
    dv_datasets = dict_vectorizer.fit_transform(datasets)
    print dv_datasets.toarray()
    print dict_vectorizer.vocabulary_
    print dict_vectorizer.feature_names_
    print "-" * 80

    #fh_vectorizer = FeatureHasher(n_features=10, input_type="dict")
    #fh_datasets = fh_vectorizer.fit_transform([{"text": 10, "words": 7}, {"name": 1, "words": 5}, {"gender": 1}])
    fh_vectorizer = FeatureHasher(n_features=10, input_type="string")
    fh_datasets = fh_vectorizer.fit_transform(["Liming love football", "Zhansan likes baseball"])
    print fh_datasets.toarray()

    raw_datasets, _ = Datasets.load_datasets()
    datasets = [v for v in raw_datasets.data[:10]]

    count_vectorizer = CountVectorizer(decode_error="ignore")
    cv_datasets = count_vectorizer.fit_transform(datasets)
    print count_vectorizer.vocabulary_

    tfidf_transformer = TfidfTransformer(smooth_idf=True)
    tfidft_datasets = tfidf_transformer.fit_transform(cv_datasets)
    print tfidft_datasets.toarray()
    print tfidf_transformer.idf_



if __name__ == "__main__":
    main()