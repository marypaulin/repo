from time import time
from math import ceil
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from lib.logger import Logger

def scalability_analysis(model_class, hyperparameters, dataset, path):
    X = dataset.values[:, :-1]
    Y = dataset.values[:, -1]
    (n, m) = X.shape
    sample_size_step = ceil(n / 20)
    feature_size_step = ceil(m / 20)

    logger = Logger(path=path, header=['samples', 'features', 'runtime'])
    for sample_size in range(1, n, sample_size_step):
        for feature_size in range(1, m, feature_size_step):
            x = X[:sample_size,:feature_size]
            y = Y[:sample_size]
            model = model_class(**hyperparameters)
            start = time()
            model.fit(x, y)
            runtime = time() - start
            logger.log([sample_size, feature_size, runtime])

def plot_scalability_analysis(dataset):
    (n, m) = dataset.shape
    x = list(set(dataset.values[:,0]))
    list.sort(x)
    y = list(set(dataset.values[:,1]))
    list.sort(y)
    Y, X = np.meshgrid(y, x)
    Z = np.array(dataset.values[:,2]).reshape(len(x), len(y))

    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Feature Size M')
    ax.set_zlabel('Runtime (s)')