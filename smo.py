"""
objective = min(1/2*sum() + b)
eta = k11 + k22 - 2*k12

Usage:
0: download  https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/
1: choose your kernrel function
2: call SmoSVM class to get your SmoSVM object
3: call SmoSVM object's fit() function
4: call SmoSVM object's predict() function
"""

import types

import numpy as np
import pandas as pd


class SmoSVM(object):
    def __init__(self, train, alpha_list, kernel_func, cost=1.0, b=0.0, tolerance=0.0, auto_norm=True):
        self.init = True
        self._auto_norm = auto_norm
        self.tags = train[:, 0]
        self.samples = self._norm(train[:, 1:]) if self._auto_norm else train[:, 1:]
        self.alphas = alpha_list
        self.Kernel = kernel_func

        self._all_samples = list(range(self.length))
        self._eps = 0.001
        self._cost = np.float64(cost)
        self._b = np.float64(b)
        self._K_matrix = self._calculate_K()
        self._error = np.zeros(self.length)
        self._tol = np.float64(tolerance) if tolerance > 0.0001 else np.float64(0.001)
        self._support = []

        self.choose_alpha = self._choose_alpha()

    # Calculate alphas using SMO algorithsm
    def fit(self):
        K = self._K
        state = None
        while True:

            # 1: Find alpha1, alpha2
            try:
                i1, i2 = self.choose_alpha.send(state)
                state = None
                # show none-obey-kkt samples' number
                # from collections import Counter
                # result = []
                # for i in self._all_samples:
                #     result.append(self._check_obey_KKT(i))
                # print(Counter(result).get(True))
            except StopIteration as e:
                print("Optimization done, every sample satisfy the KKT condition!")
                # for i in self._all_samples:
                #     if self._check_obey_KKT(i):
                #         raise ValueError('some sample not fit KKT condition')
                break

            # 2: calculate new alpha2 and new alpha1
            y1, y2 = self.tags[i1], self.tags[i2]
            s = y1 * y2
            a1, a2 = self.alphas[i1].copy(), self.alphas[i2].copy()
            E1, E2 = self._E(i1), self._E(i2)
            eta = self._get_eta(i1, i2)
            args = (eta, i1, i2, a1, a2, E1, E2, y1, y2, s)
            a1_new, a2_new = self._get_new_alpha(*args)

            if not a1_new and not a2_new:
                state = False
                continue
            self.alphas[i1], self.alphas[i2] = a1_new, a2_new

            # 3: update threshold(b)
            b1_new = np.float64(-E1 - y1 * K(i1, i1) * (a1_new - a1) - y2 * K(i2, i1) * (a2_new - a2) + self.b)
            b2_new = np.float64(-E2 - y2 * K(i2, i2) * (a2_new - a2) - y1 * K(i1, i2) * (a1_new - a1) + self.b)

            if 0.0 < a1_new < self.c:
                b = b1_new
            if 0.0 < a2_new < self.c:
                b = b2_new
            if not (np.float64(0) < a2_new < self.c) and not (np.float64(0) < a1_new < self.c):
                b = (b1_new + b2_new) / 2.0
            b_old = self.b
            self.b = b

            # 4:  update error value,here we only calculate those support vectors' error
            self.support = [i for i in self._all_samples if self.is_support(i)]
            for s in self.support:
                if s == i1 or s == i2:
                    continue
                self._error[s] += y1 * (a1_new - a1) * K(i1, s) + y2 * (a2_new - a2) * K(i2, s) + (self.b - b_old)

            # if i1 or i2 is support vector,update there error value to zero
            if self.is_support(i1):
                self._error[i1] = 0

            if self.is_support(i2):
                self._error[i2] = 0

    # Predict test samles
    def predict(self, test_samples):
        def K(index, sample):
            return self.Kernel(self.samples[index], sample)

        if test_samples.shape[1] > self.samples.shape[1]:
            raise ValueError("Test samples' feature length not equal to train samples")

        if self._auto_norm:
            test_samples = self._norm(test_samples)

        results = []
        for test_sample in test_samples:
            tag = self.tags
            alphas = self.alphas
            b = self.b
            length = self.length
            result = np.sum([alphas[j] * tag[j] * K(j, test_sample) for j in self._all_samples]) + b
            if result > 0:
                results.append(1)
            else:
                results.append(-1)

        return results

    # Check if alpha conficts with KKT condition
    def _check_obey_KKT(self, index):
        alphas = self.alphas
        tol = self.tol
        r = self._E(index) * self.tags[index]
        c = self.c
        return (r < -tol and alphas[index] < c) or (r > tol and alphas[index] > 0.0)

    # Check if sample is support vector
    def is_support(self, index):
        if 0.0 < self.alphas[index] < self.c:
            return True
        else:
            return False

    # Get value which calculated by kernel function
    def _K(self, i1, i2):
        return self._K_matrix[i1, i2]

    # Calculate Kernel matrix of all possible i1,i2 ,save time
    def _calculate_K(self):
        K_matrix = np.zeros([self.length, self.length])
        for i in self._all_samples:
            for j in self._all_samples:
                K_matrix[i, j] = np.float64(self.Kernel(self.samples[i, :], self.samples[j, :]))
        return K_matrix

    # Get sample's error 1:bound g(xi) - yi    2:none-bound _error[i]
    def _E(self, index):
        # get from error data
        if self.is_support(index):
            return self._error[index]
        # get by g(xi) - yi
        else:
            return self._predict(index) - self.tags[index]

    # Equal to g(xi)
    def _predict(self, index):
        return np.dot(self.alphas * self.tags, self._K_matrix[:, index]) + self.b

    # Get L and H which bounds the new alpha2
    def _get_LH(self, a1, a2, s):
        if s == -1:
            l, h = max(0.0, a2 - a1), min(self.c, self.c + a2 - a1)
        elif s == 1:
            l, h = max(0.0, a2 + a1 - self.c), min(self.c, a2 + a1)
        else:
            raise ValueError('s is not -1 or 1,s={}'.format(s))
        return l, h

    # Get K11 + K22 - 2*K12
    def _get_eta(self, i1, i2):
        K = self._K
        k11 = K(i1, i1)
        k22 = K(i2, i2)
        k12 = K(i1, i2)
        return k11 + k22 - 2.0 * k12

    # Get the new alpha2 and new alpha1
    def _get_new_alpha(self, eta, i1, i2, a1, a2, E1, E2, y1, y2, s):
        if i1 == i2:
            return None, None

        L, H = self._get_LH(a1, a2, s)
        if L == H:
            return None, None

        if eta > 0.0:
            a2_new_unc = a2 + (y2 * (E1 - E2)) / eta

            # a2_new has a boundry
            if a2_new_unc >= H:
                a2_new = H
            elif a2_new_unc <= L:
                a2_new = L
            else:
                a2_new = a2_new_unc

        # select the new alpha2 which could get the minimal objective
        else:
            b = self.b
            K = self._K
            L1 = a1 + s * (a2 - L)
            H1 = a1 + s * (a2 - H)

            # way 1
            f1 = y1 * (E1 + b) - a1 * K(i1, i1) - s * a2 * K(i1, i2)
            f2 = y2 * (E2 + b) - a2 * K(i2, i2) - s * a1 * K(i1, i2)
            OL = L1 * f1 + L * f2 + 1 / 2 * L1 ** 2 * K(i1, i1) + 1 / 2 * L ** 2 * K(i2, i2) + s * L * L1 * K(i1, i2)
            OH = H1 * f1 + H * f2 + 1 / 2 * H1 ** 2 * K(i1, i1) + 1 / 2 * H ** 2 * K(i2, i2) + s * H * H1 * K(i1, i2)

            # way 2
            # tmp_alphas = self.alphas.copy()
            # tmp_alphas[i1], tmp_alphas[i2] = L1, L
            # OL = self._get_objective(tmp_alphas)
            # tmp_alphas[i1], tmp_alphas[i2] = H1, H
            # OH = self._get_objective(tmp_alphas)

            if OL < (OH - self._eps):
                a2_new = L
            elif OL > OH + self._eps:
                a2_new = H
            else:
                a2_new = a2

        # a1_new has a boundry
        a1_new = a1 + s * (a2 - a2_new)
        if a1_new < 0:
            a2_new += s * a1_new
            a1_new = 0
        if a1_new > self.c:
            a2_new += s * (a1_new - self.c)
            a1_new = self.c

        return a1_new, a2_new

    # Get objective
    def _get_objective(self, alphas):
        inner_sum = lambda j: np.dot(alphas * self.tags, self._K_matrix[:, j])
        objective = 1 / 2 * (np.sum([alphas[j] * self.tags[j] * inner_sum(j) for j in self._all_samples])) - np.sum(
            alphas)

        return objective

    # Choose alpha1 and alpha2
    @types.coroutine
    def _choose_alpha(self):
        indexs = yield from self._choose_a1()
        if not indexs:
            return
        return indexs

    # Choose first alpha
    # Fisrt loop over all sample,second loop over all none-bound sample,and repeat this two process endlessly
    @types.coroutine
    def _choose_a1(self):
        while True:
            all_not_obey = True
            # all sample
            print('scanning all sample!')
            for i1 in [i for i in self._all_samples if self._check_obey_KKT(i)]:
                all_not_obey = False
                yield from self._choose_a2(i1)

            # none-bound sample
            print('scanning none-bound sample!')
            while True:
                not_obey = True
                for i1 in [i for i in self._all_samples if self._check_obey_KKT(i) and self.is_support(i)]:
                    not_obey = False
                    yield from self._choose_a2(i1)
                if not_obey:
                    print('all none-bound sample fits the KKT condition!')
                    break
            if all_not_obey:
                print('all sample fits the KKT condition.optimization done!')
                break
        return False

    # Choose the second alpha by using heuristic algorithm
    @types.coroutine
    def _choose_a2(self, i1):
        self.support = [i for i in self._all_samples if self.is_support(i)]

        if len(self.support) > 0:
            tmp_error = self._error.copy().tolist()
            tmp_error_dict = {index: value for index, value in enumerate(tmp_error) if self.is_support(index)}

            if self._E(i1) >= 0:
                i2 = min(tmp_error_dict, key=lambda index: tmp_error_dict[index])
            else:
                i2 = max(tmp_error_dict, key=lambda index: tmp_error_dict[index])

            cmd = yield i1, i2
            if cmd is None:
                return

        for i2 in np.roll(self.support, np.random.choice(self.length)):
            cmd = yield i1, i2
            if cmd is None:
                return

        for i2 in np.roll(self._all_samples, np.random.choice(self.length)):
            cmd = yield i1, i2
            if cmd is None:
                return

    # Normalise data using min_max way
    def _norm(self, data):
        # Use sklearn's normerlizer
        # from sklearn.preprocessing import MinMaxScaler
        # if self.init:
        #     self.normer = MinMaxScaler()
        #     return self.normer.fit_transform(data)
        # else:
        #     return self.normer.transform(data)

        if self.init:
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)
            self.init = False

            return (data - self._min) / (self._max - self._min)
        else:

            return (data - self._min) / (self._max - self._min)

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, value):
        self._support = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def c(self):
        return self._cost

    @property
    def tol(self):
        return self._tol

    @property
    def length(self):
        return self.samples.shape[0]


class Kernel(object):
    def __init__(self, kernel, degree=1.0, coef0=0.0, gamma=1.0):
        self.maps = {
            'linear': self._linear,
            'poly': self._polynomial,
            'rbf': self._rbf
        }
        self.degree = np.float64(degree)
        self.coef0 = np.float64(coef0)
        self.gamma = np.float64(gamma)
        self.kernel = self.maps[kernel]
        self.check()

    def _polynomial(self, v1, v2):
        return (self.gamma * np.inner(v1, v2) + self.coef0) ** self.degree

    def _linear(self, v1, v2):
        return np.inner(v1, v2) + self.coef0

    def _rbf(self, v1, v2):
        return np.exp(-1 * (self.gamma * np.linalg.norm(v1 - v2) ** 2))

    def check(self):
        if self.kernel == self._rbf:
            if self.gamma < 0:
                raise ValueError('gamma value must greater than 0')

    def __call__(self, v1, v2):
        return self.kernel(v1, v2)


def count_time(func):
    def call_func(*args, **kwargs):
        import time
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('\r\ncost {} seconds'.format(end_time - start_time))

    return call_func


@count_time
def test():
    print('hello,start test svm by smo')
    data = pd.read_csv(r'C:/Users/dell/Downloads/breast-cancer-wisconsin-data/data.csv')

    # 1: pre-processing data
    del data[data.columns.tolist()[-1]]
    del data['id']
    data = data.dropna(axis=0)
    data = data.replace({'M': np.float64(1), 'B': np.float64(-1)})
    samples = np.array(data)[:, :]

    # 2: deviding data into train data and test data
    train, test = samples[:400, :], samples[400:, :]
    test_tags, test_samples = test[:, 0], test[:, 1:]

    # 3: choose kernel function,and set alphas to zero
    mykernel = Kernel(kernel='rbf', degree=3, coef0=1, gamma=0.5)
    al = np.zeros(train.shape[0])

    # 4: calculating best alphas using SMO algorithm and predict test samples
    SVM = SmoSVM(train=train, alpha_list=al, kernel_func=mykernel, cost=0.4, b=0.0, tolerance=0.001)
    SVM.fit()
    predict = SVM.predict(test_samples)

    # 5: check accuracy
    score = 0
    for i in range(test_tags.shape[0]):
        if test_tags[i] == predict[i]:
            score += 1
    print('\r\nright: {}\r\nall: {}'.format(score, test_tags.shape[0]))
    print("Rough Accuracy: {}".format(score / test_tags.shape[0]))


if __name__ == '__main__':
    test()
