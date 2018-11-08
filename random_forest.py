"""
    Implementation of random forest.

    Class:
        RandomForest:
            Support parallel computing.
            Support calculating out-of-bag accuracy.

    Usage:
        # You need to specify your base_model!
        # Generally speaking, cart tree is used as a base_model!
        tree_args = {'n_features': feature_n}
        randomforest = RandomForest(base_model=DecisionTree, n_trees=tree_n, oob_estimate=True, model_args=tree_args)
        randomforest.fit(train_data)
        print(randomforest.oob_accuracy)
        results = randomfores.predict(test_data)

"""
import multiprocessing
import random

import numpy as np

from decision_tree import DecisionTree
from smo import SmoSVM, Kernel
from util import ModelEvaluator
from util import load_cancer_data, load_sonar_data


class RandomForest(object):

    def __init__(self, base_model, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=False, model_args={}):
        self.base_model = base_model
        self.n_trees = n_trees
        self.method = method
        self.samples_ratio = samples_ratio
        self.trained_models = []
        self.oob_estimate = oob_estimate
        self.model_args = model_args

    def fit(self, train_data):
        self.samples = train_data
        self.tags = self.samples[:, 0]
        self.length = self.samples.shape[0]
        self.samples_index = set(range(self.length))
        self.trained_models = []
        self.oob_list = []

        if self.n_trees == 1:
            sub_samples_index = self.get_sub_samples(self.samples_ratio)
            sub_samples = self.samples[sub_samples_index]
            model = self.base_model(**self.model_args).fit(sub_samples)
            self.trained_models.append(model)
            self.oob_list.append(sub_samples_index)
            return self
        # use multiprocessing lib to train decision tree in parallel.
        # another slow way: use Process and Queue,create core_num's process,1 task_queue and 1 result_queue.
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        obj_list = []
        for i in range(self.n_trees):
            sub_samples_index = self.get_sub_samples(self.samples_ratio)
            sub_samples = self.samples[sub_samples_index]
            model = self.base_model(**self.model_args)
            result = pool.apply_async(self.process, args=(model, sub_samples, sub_samples_index))
            obj_list.append(result)
        pool.close()
        pool.join()
        for obj in obj_list:
            model, sub_samples_index = obj.get()
            self.trained_models.append(model)
            self.oob_list.append(sub_samples_index)
        # estimate out-of-bag accuracy
        if self.oob_estimate:
            self.oob_accuracy, self.oob_error = self.predict_oob()

        return self

    def predict(self, test_samples):
        if test_samples.shape[1] != (self.samples.shape[1] - 1):
            raise ValueError('Feature numbers not equal\r\nTrain:{} Test:{}'.format(self.samples.shape[1] - 1,
                                                                                    test_samples.shape[1]))
        results = []
        for sample in test_samples:
            predictions = [model.predict([sample])[0] for model in self.trained_models]
            calculated_prediction = self.handle_predictions(predictions)
            results.append(calculated_prediction)

        return results

    # process in parallel.
    def process(self, model, sub_samples, sub_samples_index):
        trained_model = model.fit(sub_samples)
        return trained_model, sub_samples_index

    # predict out-of-bag accuracy.
    def predict_oob(self):
        predicted_tags = []
        true_tags = []
        for sample_index in self.samples_index:
            tmp_list = []
            for model_index in range(len(self.trained_models)):
                if sample_index not in self.oob_list[model_index]:
                    sample = [np.array(self.samples[sample_index][1:])]
                    tmp_list.append(self.trained_models[model_index].predict(sample)[0])
            if tmp_list:
                voted_tag = max(((tag, tmp_list.count(tag)) for tag in set(tmp_list)), key=lambda x: x[1])[0]
                predicted_tags.append(voted_tag)
                true_tags.append(self.tags[sample_index])

        accuracy = (np.array(predicted_tags) - true_tags).tolist().count(0) / len(true_tags)
        error = 1 - accuracy
        return accuracy, error

    # handle prediction generated from decison trees,use vote method in normal situations.
    def handle_predictions(self, predictions):
        if self.method == 'vote':
            counter = ((tag, predictions.count(tag)) for tag in set(predictions))
            final_prediction = max(counter, key=lambda x: x[1])[0]
        else:
            raise ValueError

        return final_prediction

    # random forest requires random samples get by bootstrap for each decision tree.
    def get_sub_samples(self, ratio):
        samples_num = round(self.length * ratio)
        sub_samples_index = []
        for j in range(samples_num):
            sub_samples_index.append(random.randrange(self.length))
        return sub_samples_index


def test_randomforest_cart():
    print('test cart random forest using cancer data')
    data = load_cancer_data()
    tree_args = {'criterion': 'gini', 'n_features': 7, 'max_depth': 15, 'min_impurity': 1e-6}
    rf = RandomForest(DecisionTree, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,
                      model_args=tree_args)
    rf.fit(data)
    print('out-of-bag accuracy is :{}'.format(rf.oob_accuracy))

    print('test cart random forest using sonar data')
    data = load_sonar_data()
    tree_args = {'criterion': 'gini', 'n_features': 7, 'max_depth': 15, 'min_impurity': 1e-6}
    rf = RandomForest(DecisionTree, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,
                      model_args=tree_args)
    rf.fit(data)
    print('out-of-bag accuracy is :{}'.format(rf.oob_accuracy))
    return


def test_randomforest_svm():
    print('test svm random forest using cancer data')
    data = load_cancer_data()
    mykernel = Kernel(kernel='poly', degree=5, coef0=1, gamma=0.5)
    svm_args = {'kernel_func': mykernel, 'cost': 0.4, 'b': 0.0, 'tolerance': 0}
    rf = RandomForest(SmoSVM, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,
                      model_args=svm_args)
    rf.fit(data)
    print('out-of-bag accuracy is :{}'.format(rf.oob_accuracy))

    print('test svm random forest using sonar data')
    data = load_sonar_data()
    mykernel = Kernel(kernel='poly', degree=5, coef0=1, gamma=0.5)
    svm_args = {'kernel_func': mykernel, 'cost': 0.4, 'b': 0.0, 'tolerance': 0}
    rf = RandomForest(base_model=SmoSVM, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,
                      model_args=svm_args)
    rf.fit(data)
    print('out-of-bag accuracy is :{}'.format(rf.oob_accuracy))
    return


def evaluate_randomforest_cart():
    data = load_sonar_data()
    tree_args = {'criterion': 'gini', 'n_features': 14, 'max_depth': 20, 'min_impurity': 1e-6}
    rf_args = {'base_model': DecisionTree, 'n_trees': 100, 'method': 'vote', 'samples_ratio': 1.0,
               'oob_estimate': False,
               'model_args': tree_args}
    result = ModelEvaluator(model=RandomForest, samples=data, method='cross', fold=10, model_args=rf_args).evaluate()
    print(result)
    return


def evaluate_randomforest_svm():
    data = load_sonar_data()
    mykernel = Kernel(kernel='poly', degree=5, coef0=1, gamma=0.5)
    svm_args = {'kernel_func': mykernel, 'cost': 0.4, 'b': 0.0, 'tolerance': 0}
    rf_args = {'base_model': SmoSVM, 'n_trees': 100, 'method': 'vote', 'samples_ratio': 1.0, 'oob_estimate': False,
               'model_args': svm_args}
    result = ModelEvaluator(model=RandomForest, samples=data, method='cross', fold=10, model_args=rf_args).evaluate()
    print(result)
    return


if __name__ == '__main__':
    test_randomforest_cart()
    test_randomforest_svm()
    evaluate_randomforest_cart()
    evaluate_randomforest_svm()
