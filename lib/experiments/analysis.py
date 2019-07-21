import numpy as np
import matplotlib.pyplot as plt

from gc import collect
from time import time, sleep
from math import ceil, floor
from mpl_toolkits import mplot3d
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from lib.experiments.logger import Logger
from lib.data_structures.dataset import read_dataframe

def accuracy_analysis(dataset, model_class, hyperparameters, path):
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1]

    # Perform cross validation over k-folds, one for each proposed hyperparameter
    if len(hyperparameters) == 1:
        hyperparameters = [hyperparameters[0] for _i in range(2)]
    kfolds = KFold(n_splits=len(hyperparameters))

    logger = Logger(path=path, header=['hyperparameter','width', 'accuracy'])

    model_index = 0
    for train_index, test_index in kfolds.split(X):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        hyperparameter = hyperparameters[model_index]

        model = model_class(**hyperparameter)

        try:
            model.fit(X_train, y_train)
        except:
            pass
        else:
            accuracy = model.score(X_test, y_test)

            # Compute Tree Width of the model
            if model_class == DecisionTreeClassifier:
                width = compute_width(model)
            else:
                width = model.width

            logger.log([str(hyperparameter), width, accuracy])
        model_index += 1

def plot_accuracy_analysis(dataset_name, title):
    fig = plt.figure(figsize=(10, 8), dpi=100)

    dataset = read_dataframe('data/accuracy/{}/{}.csv'.format(dataset_name, 'cart'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [ accuracies[width] for width in x ]
    plt.plot(x, y, label='cart', markersize=5, marker='o', linewidth=0)

    dataset = read_dataframe('data/accuracy/{}/{}.csv'.format(dataset_name, 'osdt'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [ accuracies[width] for width in x ]
    plt.plot(x, y, label='osdt', markersize=5, marker='o', linewidth=0)

    dataset = read_dataframe('data/accuracy/{}/{}.csv'.format(dataset_name, 'parallel_osdt'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [ accuracies[width] for width in x ]
    plt.plot(x, y, label='parallel_osdt', markersize=5, marker='o', linewidth=0)

    plt.xlabel('Tree Width')
    plt.ylabel('Test Accuracy')
    plt.grid()
    plt.legend()
    plt.title(title)

def scalability_analysis(dataset, model_class, hyperparameters, path, step_count=10):
    X = dataset.values[:, :-1]
    Y = dataset.values[:, -1]
    (n, m) = X.shape
    sample_size_step = max(1, round(n / step_count))
    feature_size_step = max(1, round(m / step_count))

    logger = Logger(path=path, header=['samples', 'features', 'runtime'])

    for sample_size in range(1, n+1, sample_size_step):
        for feature_size in range(1, m+1, feature_size_step):
            print("Subsample Shape: ({}, {})".format(sample_size, feature_size))

            # Try to standardize starting state
            collect()
            sleep(1)

            # Take Subsample
            x = X[:sample_size,:feature_size]
            y = Y[:sample_size]

            model = model_class(**hyperparameters)
            start = time()
            reruns = 5
            runtimes = []
            for i in range(reruns):
                try:
                    model.fit(x, y)
                except Exception as e:
                    print(e)
                    pass
                runtime = time() - start
                runtimes.append(runtime)
            
            list.sort(runtimes)
            runtime = runtimes[floor(reruns/2)]
            logger.log([sample_size, feature_size, runtime])

def plot_scalability_analysis(dataset, title, z_limit=None):
    (n, m) = dataset.shape
    x = list(set(dataset.values[:,0]))
    list.sort(x)
    y = list(set(dataset.values[:,1]))
    list.sort(y)
    Y, X = np.meshgrid(y, x)
    Z = np.array(dataset.values[:,2]).reshape(len(x), len(y))

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Feature Size M')
    ax.set_zlabel('Runtime (s)')
    ax.set_title(title)

    if z_limit != None:
        ax.set_zlim(0, z_limit)

    ax.view_init(50, -20)
    
# Parses the DecisionTreeClassifier from Sci-Kit Learn according to their documentation
# Returns the number of leaves in this model
# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def compute_width(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    leaf_count = 0
    for i in range(n_nodes):
        if is_leaves[i]:
            leaf_count += 1
    return leaf_count

# Overview: Module containing functions for hyerparameter optimization

# Summary: Train and select the best model over a list of different hyperparameter settings using cross validation
# Input:
#    dataset: Pandas dataframe containing n-1 columns of features followed by 1 column of labels
#    model_class: Python class implementing standard sci-kit-learn model interface as follows
#       __init__(self, hyperparameters)
#       fit(self, X_train, y_test)
#       score(X_test, y_test)
#   hyperparameters: list of dictionaries each containing keyword arguments holding hyperparameter assignments for model construction
#   retrain: flag to retrain on the full dataset using the optimal hyperparameters
# Output:
#   model: the model that scored the highest in test accuracy during cross-validation
#   accuracy: the test accuracy of the model that scored the highest
#   hyperparameter: the hyperparameter setting that resulted in the highest test accuracy

def train_cross_validate(dataset, model_class, hyperparameters=[{}], retrain=False):
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1]

    # Perform cross validation over k-folds, one for each proposed hyperparameter
    if len(hyperparameters) == 1:
        hyperparameters = [hyperparameters[0] for _i in range(2)]
    kfolds = KFold(n_splits=len(hyperparameters))

    model_index = 0
    optimal_hyperparameter = None
    optimal_model = None
    optimal_accuracy = 0
    for train_index, test_index in kfolds.split(X):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        hyperparameter = hyperparameters[model_index]

        model = model_class(**hyperparameter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        if accuracy >= optimal_accuracy:
            optimal_hyperparameters = hyperparameter
            optimal_model = model
            optimal_accuracy = accuracy

        model_index += 1

    # Note: This retrains the model over the full dataset, which breaks the association with the test accuracy
    if retrain == True:
        optimal_model = model_class(**optimal_hyperparameter)
        optimal_model.fit(X, y)

    return optimal_model, optimal_accuracy, optimal_hyperparameter
