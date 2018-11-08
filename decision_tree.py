"""
    Implementation of decision tree.

    Class:
        DecisonTree:
            Cart tree (binary tree).
            Support giniindex and information gain.
        DiscreteTree:
            Tree for discrete variables.
            Support ID3 (information gain).
            Support C4.5 (information gain ratio).
            Not tested!

    Usage:
        cart_tree = DecisionTree(criterion='gini', n_features=None, max_depth=20, min_impurity=None)
        cart_tree.fit(train_data)
        results = cart.predict(test_data)

"""
from __future__ import division

import random
from math import log

import numpy as np

from util import ModelEvaluator
from util import load_sonar_data, load_cancer_data


class Node(object):
    def __init__(self, depth, feature, value, tag, index):
        self.depth = depth
        self.feature = feature
        self.value = value
        self.tag = tag
        self.index = index
        self.children = None

    def __str__(self):
        return '{} {} {} \r\n{}'.format(self.feature, self.value, self.tag, self.children)


class DecisionTree(object):
    def __init__(self, criterion='gini', n_features=None, max_depth=20, min_impurity=1e-6):
        self.criterion = criterion
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, train_data):
        self.data = train_data
        self.samples = self.data[:, 1:]
        self.tags = self.data[:, 0]
        self.tree = self.get_tree()

        return self

    def predict(self, test_samples):
        results = []
        for sample in test_samples:
            results.append(self.scan_tree(sample, self.tree))
        return results

    # for every node ,use it's value to choose searching path.
    def scan_tree(self, sample, node):
        # if one node has no children,then searching process stop,return it's tag as predicted tag.
        if node.children is None:
            return node.tag

        left_node, right_node = node.children[0], node.children[1]
        decision_feature = left_node.feature
        if sample[decision_feature] <= left_node.value:
            selected_node = left_node
        else:
            selected_node = right_node

        return self.scan_tree(sample, selected_node)

    # create root node and split this node.
    def get_tree(self):
        full_index = np.arange(len(self.tags))
        tag = self.find_tag(full_index)
        root_node = Node(depth=1, feature=None, value=None, tag=tag, index=full_index)
        full_features = list(range(self.samples.shape[1]))
        root_node.children = self.split(root_node, full_features)

        return root_node

    def split(self, node, features):
        if self.check_stop(node, features):
            return None
        # find best split feature and feature-value based on ginindex or information gain.
        feature, value = self.find_best_point(node, features)
        # create left node using samples which feature value lower than feature value and create right node vise versa.
        left_index, right_index = self.split_left_right(node.index, feature, value)
        left_tag = self.find_tag(left_index)
        right_tag = self.find_tag(right_index)
        nodes = [Node(node.depth + 1, feature, value, left_tag, left_index, ),
                 Node(node.depth + 1, feature, value, right_tag, right_index)]
        # remove this feature from available features list.
        if feature in features:
            features.remove(feature)
        # split for each child node too.
        for node in nodes:
            node.children = self.split(node, features)

        return nodes

    def find_best_point(self, node, features):
        index = node.index
        features = self.get_sub_features(features)
        feature, value = None, None
        min_giniindex = 1000
        max_gain = -1000
        # find min giniindex
        if self.criterion == 'gini':
            for _feature in features:
                values_list = set(self.samples[:, _feature][index])
                for _value in values_list:
                    gini_index = self.gini_index(index, _feature, _value)
                    if gini_index < min_giniindex:
                        min_giniindex = gini_index
                        feature = _feature
                        value = _value
        # find max information gain
        else:
            for _feature in features:
                values_list = list(set(self.samples[:, _feature][index]))
                for _value in values_list:
                    gain = self.gain(index, _feature, _value)
                    if gain > max_gain:
                        max_gain = gain
                        feature = _feature
                        value = _value

        return feature, value

    # a normal tree find best split point from remained available features.
    # randomForest need to select best split point within a random subset of available features.
    def get_sub_features(self, features):
        if self.n_features is not None and self.n_features <= len(features):
            features = random.sample(features, self.n_features)

        return features

    def check_stop(self, node, features):
        # stop split when there is no feature to split node.
        if len(features) == 0:
            return True
        # stop split when there is only one class in this node.
        if len(set(self.tags[node.index])) == 1:
            return True
        # stop split when child nodes meet maximum depth.
        if self.max_depth and (node.depth + 1) > self.max_depth:
            return True
        # stop split when raech min impurity.
        if self.min_impurity is not None:
            if self.criterion == 'gini':
                if self.gini(node.index) < self.min_impurity:
                    return True
            else:
                if self.ent(node.index) / 2 < self.min_impurity:
                    return True
        # normal, continue split this node into child nodes.
        return False

    def find_tag(self, index):
        tags = self.tags[index].tolist()
        counter = ((tag, tags.count(tag)) for tag in set(tags))
        return max(counter, key=lambda x: x[1])[0]
        # another slow way: return sorted(tags, key=tags.count, reverse=True)[0]

    def split_left_right(self, index, feature, feature_value):
        feature_values = self.samples[:, feature]
        left_index, right_index = [], []
        for _index in index:
            if feature_values[_index] <= feature_value:
                left_index.append(_index)
            else:
                right_index.append(_index)
        # another slow way 1: [i for i in index if values[i] <= 80],then use set
        # another slow way 2: np.where((array <= 80)==True),then use set
        return left_index, right_index

    def gini_index(self, index, feature, feature_value):
        left_index, right_index = self.split_left_right(index, feature, feature_value)
        sums = len(index)
        gini_index = (len(left_index) / sums) * self.gini(left_index) \
                     + (len(right_index) / sums) * self.gini(right_index)

        return gini_index

    def gini(self, index):
        tags = self.tags[index].tolist()
        sums = len(tags)
        gini = 1
        # another slow way: use collection's Counter.
        for tag in set(tags):
            gini -= (tags.count(tag) / sums) ** 2
        return gini

    def gain(self, index, feature, feature_value):
        left_index, right_index = self.split_left_right(index, feature, feature_value)
        hd = self.ent(index)
        sums = len(index)
        hda = ((len(left_index) / sums) * self.ent(left_index)
               + (len(right_index) / sums) * self.ent(right_index))
        gain = hd - hda
        return gain

    def ent(self, index):
        tags = self.tags[index].tolist()
        sums = len(tags)
        ent = 0
        for tag in set(tags):
            prob = tags.count(tag) / sums
            ent += -1 * prob * log(prob, 2)
        return ent


class DiscreteTree(DecisionTree):
    """
        Not tested!
        Implemented ID3 and C4.5 algorithm.
    """

    def __init__(self, *args):
        super().__init__(*args)
        if self.criterion not in ('cd3', 'c45'):
            raise ValueError('please set criterion to cd3 or c45')

    def scan_tree(self, sample, node):
        # if one node has no children,then searching process stop,return it's tag as predicted tag.
        if node.children is None:
            return node.tag
        decision_feature = node.children[0].feature
        selected_node = None
        # find the equal feature value in child nodes.
        for child_node in node.children:
            if sample[decision_feature] == child_node.value:
                selected_node = left_node
                break
        if selected_node is None:
            raise ValueError('feature value not exist in train data')

        return self.scan_tree(sample, selected_node)

    # crete child nodes using samples equal to every feature value.
    def split(self, node, features):
        if self.check_stop(node, features):
            return None
        # find best split feature and feature-value based on information gain and information gain ratio.
        feature, values = self.find_best_point(node, features)
        # create dict containg key:value representing each value and indexs which feature values equal to this value.
        feature_values = self.samples[:, feature]
        value_index_dict = {value: [] for value in values}
        for _index in node.index:
            for value in values:
                if feature_values[_index] == value:
                    value_index_dict[value].append(_index)
                    break

        # create many child nodes.
        nodes = []
        for value, value_index in value_index_dict.items():
            tag = self.find_tag(value_index)
            _node = Node(node.depth + 1, feature, value, tag, value_index)
            nodes.append(_node)

        # remove this feature from available features list.
        if feature in features:
            features.remove(feature)
        # split for each child node too.
        for node in nodes:
            node.children = self.split(node, features)

        return nodes

    def gain(self, index, feature, feature_value):
        hd = self.ent(index)
        feature_values = self.samples[:, feature]
        values_dict = {}
        sums = 0
        for f_index, feature_value in enumerate(feature_values):
            if f_index not in index:
                continue
            index_li = values_dict.setdefault(feature_value, [])
            index_li.append(f_index)
            sums += 1
        gain = hd
        for feature_value, index_li in values_dict.items():
            portion = len(index_li) / sums
            gain -= portion * self.ent(index_li)

        return gain

    def gain_ratio(self, index, feature, feature_value):
        gain = self.gain(index, feature, None)
        feature_values = self.samples[:, feature][index]
        counter = Counter(feature_values)
        sums = len(feature_values)
        iv = 0
        for feature_value in counter.keys():
            portion = counter[feature_value] / sums
            iv -= portion * log(portion, 2)
        gain_ratio = gain / iv
        return gain_ratio

    def find_best_point(self, node, features):
        # another slow way: map(lambda _feature: [_feature, self.gain(index, _feature)], features)
        if self.criterion == 'id3':
            feature_gain_list = ((_feature, self.gain(index, _feature, None)) for _feature in features)
        elif self.criterion == 'c45':
            feature_gain_list = ((_feature, self.gain_ratio(index, _feature, None)) for _feature in features)
        else:
            raise ValueError('criteria {} not support'.format(self.criterion))

        feature, max_gain = max(feature_gain_list, key=lambda feature_gain: feature_gain[1])
        values = list(set(self.samples[:, features][index]))

        return feature, values


def test_decisiontree():
    data = load_sonar_data()
    train, test = data[:150, :], data[150:, :]
    tree = DecisionTree(criterion='gini', n_features=None, max_depth=20, min_impurity=1e-6)
    tree.fit(train)
    predicted_tags = tree.predict(test[:, 1:])
    test_tags = test[:, 0]
    accuracy = list(predicted_tags - test_tags).count(0) / len(test_tags)
    print(accuracy)


def evaluate_decisiontree():
    print('test cancel data')
    data = load_cancer_data()
    tree_args = {'criterion': 'gini', 'n_features': 7, 'max_depth': 15, 'min_impurity': 1e-6}
    result = ModelEvaluator(model=DecisionTree, samples=data, method='cross', fold=10, model_args=tree_args).evaluate()
    print(result)
    print('test sonar data')
    data = load_sonar_data()
    tree_args = {'criterion': 'gini', 'n_features': 7, 'max_depth': 15, 'min_impurity': 1e-6}
    result = ModelEvaluator(model=DecisionTree, samples=data, method='cross', fold=10, model_args=tree_args).evaluate()
    print(result)
    return


if __name__ == '__main__':
    test_decisiontree()
    evaluate_decisiontree()
