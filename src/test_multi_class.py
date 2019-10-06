import pandas as pd
import numpy as np
import heapq
import math
import time
from sklearn.model_selection import train_test_split
import gmpy2
from gmpy2 import mpz
import re
from sklearn import tree

from osdt_multi_class import bbound, predict


def test_multi(file, lambs, timelimit=1800):
    count = 1

    for lamb in lambs:
        for i in range(1, 11):
            print(count)
            data_train = pd.DataFrame(pd.read_csv(file, sep=","))

            X = data_train.values[:, :-1]
            y = data_train.values[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
            # CART
            clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=max(math.ceil(lamb * 2 * len(y_train)), 2),
                                              min_samples_leaf=math.ceil(lamb * len(y_train)),
                                              max_leaf_nodes=math.floor(1 / (2 * lamb)),
                                              min_impurity_decrease=lamb
                                              )
            clf = clf.fit(X_train, y_train)
            nleaves_CART = (clf.tree_.node_count + 1) / 2
            trainaccu_CART = clf.score(X_train, y_train)
            print("CART Objective score:",1 - trainaccu_CART + lamb*nleaves_CART)
            print("CART Training Accuracy:", trainaccu_CART)
            print("Number of CART leaves:", nleaves_CART)
            testaccu_CART = clf.score(X_test, y_test)

            #OSDT
            y_train = pd.get_dummies(y_train).values
            leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT = \
                bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=False)

            print("Number of OSDT leaves:", nleaves_OSDT)
            print("CART Test Accuracy:", testaccu_CART)
            yhat, accu = predict(leaves_c, prediction_c, dic, X_test, y_test)
            count += 1
            # 

lambs1 = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025]
#test_multi('../data/preprocessed/monk1_multi_small.csv', lambs=0.025)
#test_multi('../data/preprocessed/monk1_very_small.csv', lambs=0.025)
#test_multi('../data/preprocessed/car_multi', lambs=0.025)

test_multi('../data/multi-class/iris-multi-class.csv', lambs=lambs1)

