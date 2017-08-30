# -*- coding:utf-8 -*-

import jieba
import jieba.analyse as analyse

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_datasets(filename):
    ori_datasets = []
    seg_datasets = []
    with open(filename, "rb") as f:
        for line in f:
            seg_datasets.append(" ".join(jieba.lcut(line.strip())))
            ori_datasets.append(line.strip())

    return ori_datasets, seg_datasets


def output(ori_datasets, seg_datasets, labels):
    clusters = {}
    clusters_ori_datasets = {}
    clusters_seg_datasets = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = ""
        clusters[label] += " " + seg_datasets[idx]
        if label not in clusters_ori_datasets:
            clusters_ori_datasets[label] = []
        clusters_ori_datasets[label].append(ori_datasets[idx])
        if label not in clusters_seg_datasets:
            clusters_seg_datasets[label] = []
        clusters_seg_datasets[label].append(seg_datasets[idx])

    for idx, label in enumerate(clusters):
        key_words = " ".join(analyse.textrank(clusters[label], topK=10, allowPOS=
            ["a", "ng", "n", "nr", "ns", "nt", "nz"]))
        print "cluster: %d\tset size: %d\tkey words: %s" % (idx + 1, len(clusters_ori_datasets[label]), key_words)
        for ii in range(len(clusters_ori_datasets[label])):
            print "\t\t\t\t", clusters_ori_datasets[label][ii], "||", clusters_seg_datasets[label][ii]

def main():
    ori_datasets, seg_datasets = load_datasets("./data/comments.txt")
    vectorizer = HashingVectorizer(n_features=2**15, decode_error="ignore")
    array = vectorizer.fit_transform(seg_datasets).toarray()
    print array.shape
    pca = PCA(n_components=min(array.shape[0], array.shape[1]), svd_solver='randomized')
    pca_array = pca.fit_transform(array)

    cluster = KMeans(n_clusters=20)
    model = cluster.fit(pca_array)
    output(ori_datasets, seg_datasets, cluster.labels_)


if __name__ == '__main__':
    main()