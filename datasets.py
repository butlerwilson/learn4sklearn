#-*- coding:utf-8 -*-

from sklearn.datasets import load_files

TRAIN_DATASETS_DIR = "./data/20news-bydate-train"
TEST_DATASETS_DIR = "./data/20news-bydate-test"

class Datasets(object):
    @staticmethod
    def load_datasets():
        train_datasets = load_files(TRAIN_DATASETS_DIR)
        test_datasets = load_files(TEST_DATASETS_DIR)

        return train_datasets, test_datasets