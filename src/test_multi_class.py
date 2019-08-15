import pandas as pd
import numpy as np
import heapq
import math
import time

import gmpy2
from gmpy2 import mpz
import re
from sklearn import tree

from osdt import bbound, predict


def test_multi(file, lambs, timelimit=1800):
    data_train = pd.DataFrame(pd.read_csv(file, sep=";"))
    X_train = data_train.values[:, :-1]
    y_train = data_train.values[:, -1]

    # CART
    clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=max(math.ceil(lambs * 2 * len(y_train)), 2),
                                              min_samples_leaf=math.ceil(lambs * len(y_train)),
                                              max_leaf_nodes=math.floor(1 / (2 * lambs)),
                                              min_impurity_decrease=lambs
                                              )

    clf = clf.fit(X_train, y_train)
    nleaves_CART = (clf.tree_.node_count + 1) / 2
    trainaccu_CART = clf.score(X_train, y_train)
    print(1 - trainaccu_CART + lambs*nleaves_CART)
    print(nleaves_CART)
    #OSDT
    y_train = pd.get_dummies(y_train).values
    leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT, best_is_cart= \
        bbound(X_train, y_train, lamb=lambs, prior_metric="curiosity", timelimit=timelimit, init_cart=False)

    print(nleaves_OSDT)

#test_multi('../data/preprocessed/monk1_multi_small.csv', lambs=0.025)
test_multi('../data/preprocessed/monk1_very_small.csv', lambs=0.025)