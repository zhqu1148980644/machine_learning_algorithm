# Machine Learning Algorithm implemented by python

## 1:SMO(Sequential Minimal Optimization)
### Usage:
	from smo import SmoSVM, Kernel
	
	kernel = Kernel(kernel='poly', degree=3, coef0=1, gamma=0.5)
    init_alphas = np.zeros(train.shape[0])
	SVM = SmoSVM(alpha_list=init_alphas, kernel_func=kernel, cost=0.4, b=0.0, tolerance=0.001)
    SVM.fit(train_data)
    predict = SVM.predict(test_samples)
	
### Output:
![smo](other_file/smo.png)

## 2:Decision Tree
### Usage:
    from decision_tree import DecisionTree
    
    tree = DecisionTree(criterion='gini', n_features=None, max_depth=20, min_impurity=1e-6)
    tree.fit(train_data)
    prediction = tree.predict(test_data)
    
## 3:Random Forest
### Usage:
    from decision_tree import DecisionTree
    from random_forest import RandomForest
    from smo import Kernel, SmoSVM
    
    # use cart tree
    tree_args = {'criterion': 'gini', 'n_features': 7, 'max_depth': 15, 'min_impurity': 1e-6}
    rf = RandomForest(DecisionTree, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,model_args=tree_args)
    rf.fit(train_data)
    prediction = rf.predict(test_data)
    print(rf.oob_accuracy)
    # use svm
    mykernel = Kernel(kernel='rbf', degree=5, coef0=1, gamma=0.5)
    svm_args = {'kernel_func': mykernel, 'cost': 0.4, 'b': 0.0, 'tolerance': 0}
    rf = RandomForest(SmoSVM, n_trees=100, method='vote', samples_ratio=1.0, oob_estimate=True,model_args=svm_args)
    rf.fit(train_data)
    prediction = rf.predict(test_data)

## 4:Louvain algorith(for community detection)
### Usage:
    # Most parts are adopted from python-louvain package in pypi.
    # Leiden algorithm which is an extension of Louvain algorithm is recommended.
    # See https://github.com/vtraag/leidenalg
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.erdos_renyi_graph(100, 0.05, seed=2)
    louvain = Louvain1(G, res=1.0, random_state=1)
    part = louvain.split()
    partitions = louvain.partitions
    level = 0
    for _partition in partitions:
        print("level:{} ".format(level), _partition)
        level += 1
    print("final partition: ", part)