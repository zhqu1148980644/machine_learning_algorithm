# Machine Learning Algorithm implemented by python

## 1:SMO(Sequential Minimal Optimization)
### Usage:
	from smo import SmoSVM, Kernel
	
	kernel = Kernel(kernel='poly', degree=3, coef0=1, gamma=0.5)
    init_alphas = np.zeros(train.shape[0])
	SVM = SmoSVM(train=train, alpha_list=init_alphas, kernel_func=kernel, cost=0.4, b=0.0, tolerance=0.001)
    SVM.fit()
    predict = SVM.predict(test_samples)
	
### Test:
	python3 smo.py
	
### Output:
![smo](other_file/smo.png)

#### Reference:
> [Fast Training of Support Vector Machinesusing Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf)

> [Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

> [Sequential Minimal Optimization for SVM](http://web.cs.iastate.edu/~honavar/smo-svm.pdf)

> [Implementing a Support Vector Machine using Sequential Minimal Optimization and Python 3.5](https://jonchar.net/notebooks/SVM/)


