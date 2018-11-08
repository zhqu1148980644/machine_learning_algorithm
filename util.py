"""
    Support basic tools for other algorithsm

    1: count time:
        @count_cls()
        @count
    2: load data:
        load_cancer_data()
        load_sonar_data()
    3: Evaluate model,support cross-validation,bootstrap-validation,hold-out-validation
        ModelEvaluator().evaluate()


"""
import os
import random
import sys
import urllib

import numpy as np
import pandas as pd
from line_profiler import LineProfiler

CANCER_DATASET_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
SONAR_DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def count_cls():
    def func_line_time(f):
        def decorator(self, *args, **kwargs):
            func_return = f(self, *args, **kwargs)
            lp = LineProfiler()
            lp_wrap = lp(f)
            lp_wrap(self, *args, **kwargs)
            lp.print_stats()
            return func_return

        return decorator

    return func_line_time


def count(func):
    def decorator(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('cost:{}'.format(end - start))
        return result

    return decorator


def load_cancer_data():
    data = find_data('cancel_data', CANCER_DATASET_URL)
    del data[data.columns.tolist()[0]]
    data = data.dropna(axis=0)
    data = data.replace({'M': np.float64(1), 'B': np.float64(-1)})
    data = np.array(data)[:, :]

    return data


def load_sonar_data():
    data = find_data('sonar_data', SONAR_DATASET_URL)
    data = data.replace({'M': -1, 'R': 1})
    data.insert(0, 'tag', data.iloc[:, -1])
    del data[data.columns[-1]]
    data = np.array(data)[:, :]

    return data


def find_data(name, url):
    if not os.path.exists(r'other_file/{}.csv'.format(name)):
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
        )
        response = urllib.request.urlopen(request)
        content = response.read().decode('utf-8')
        if not os.path.exists('other_file'):
            os.mkdir(r'other_file')
        with open(r'other_file/{}.csv'.format(name), 'w') as f:
            f.write(content)

    data = pd.read_csv(r'other_file/{}.csv'.format(name), header=None)
    return data


class ModelEvaluator(object):

    def __init__(self, model, samples, method='cross', fold=10, model_args={}):
        self.model = model
        self.evaluator = self.get_evaluator(method)
        self.fold = fold
        self.model_args = model_args

        self.samples = samples
        self.tags = self.samples[:, 0]
        self.length = self.samples.shape[0]
        self.trained_subsamples = []

    def evaluate(self):
        tmp_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        train_test_list = self.evaluator()
        results = []
        for train_index, test_index in train_test_list:
            self.trained_subsamples.append(train_index)
            train = self.samples[train_index]
            test = self.samples[test_index]
            model = self.model(**self.model_args).fit(train)
            test_tags, predicted_tags = test[:, 0], model.predict(test[:, 1:])

            accuracy = (test_tags - predicted_tags).tolist().count(0) \
                       / len(test_tags)
            results.append(accuracy)

        sys.stdout = tmp_stdout
        return np.mean(results)

    # train and test dataset could be generated from several sampling method.
    def get_evaluator(self, method):
        maps = {
            'cross': self.cross_evaluate,
            'holdout': self.holdout_evaluate,
            'bootstrap': self.bootstrap_evaluate
        }
        evaluator = maps.get(method, None)
        if evaluator is None:
            raise ValueError('Evaluation method {} not support'.format(method))
        return evaluator

    # cross validation
    def cross_evaluate(self):
        fold = self.fold
        if fold > self.length or fold < 2:
            raise ValueError("fold can only be set within the range of 2 to sample's length")
        subsample_length = int(self.length / fold)
        allsamples = list(range(self.length))
        subsamples_list = []
        # k fold
        for subsample_index in range(fold):
            subsample = []
            for i in range(subsample_length):
                index = random.randrange(len(allsamples))
                subsample.append(allsamples.pop(index))
            subsamples_list.append(subsample)

        # all possible combinations
        for test in subsamples_list:
            tmp = subsamples_list.copy()
            tmp.remove(test)
            train = sum(tmp, [])
            yield train, test

    # split data into 70%train and 30%test.
    def holdout_evaluate(self):
        tags = set(self.tags)
        tag_nums = len(tags)
        tags_list = list(tags)
        tags_index_list = [[] for i in tags]
        for tag_index in range(self.length):
            for i in range(tag_nums):
                if self.tags[tag_index] == tags_list[i]:
                    tags_index_list[i].append(tag_index)
                    break
        for fold_index in range(self.fold):
            train, test = [], []
            for i in range(tag_nums):
                samples_index = random.sample(tags_index_list[i], len(tags_index_list[i]))
                train_length = int(len(samples_index) * 0.7)
                train = sum([train, samples_index[:train_length]], [])
                test = sum([test, samples_index[train_length:]], [])
            yield train, test

    # bootstrap validation.
    def bootstrap_evaluate(self):
        allsamples = set(range(self.length))
        for fold_index in range(self.fold):
            train = []
            for i in range(self.length):
                index = random.randrange(self.length)
                train.append(index)
            yield train, allsamples - set(train)


load_sonar_data()
