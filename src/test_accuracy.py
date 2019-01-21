import pandas as pd
import numpy as np
import heapq
import math
import time

import gmpy2
from gmpy2 import mpz
import re

from sklearn import tree

import cProfile

from random import randint, sample, seed

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold # import KFold

from corels_dt_nosimilar_multicopies import bbound, predict

# Read in the dataset
compas = pd.DataFrame(pd.read_csv('../data/preprocessed/compas-binary.csv',sep=";"))
monk1 = pd.DataFrame(pd.read_csv('../data/preprocessed/monk1-train.csv',sep=";"))
monk2 = pd.DataFrame(pd.read_csv('../data/preprocessed/monk2-train.csv',sep=";"))
monk3 = pd.DataFrame(pd.read_csv('../data/preprocessed/monk3-train.csv',sep=";"))
balance = pd.DataFrame(pd.read_csv('../data/preprocessed/balance-scale.csv',sep=";"))
tictactoe = pd.DataFrame(pd.read_csv('../data/preprocessed/tic-tac-toe.csv',sep=";"))
car = pd.DataFrame(pd.read_csv('../data/preprocessed/car-evaluation.csv',sep=";"))


def test_accuracy(file, lambs, file_CART, file_OSDT, timelimit):
    """
    Run CART and OSDT
    split data into 3 folds, with 2 folds to train, 1 fold to test
    :param X:
    :param y:
    :param lambs:
    :param file_CART:
    :param file_OSDT:
    :return:
    """
    with open(file_CART, 'a+') as f:
        f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_CART", "testaccu_CART"]) + '\n')
    with open(file_OSDT, 'a+') as f:
        f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_CART", "testaccu_CART", "totaltime", "time_c", "leaves_c"]) + '\n')
    for lamb in lambs:
        for i in range(1, 4): # 3 folds

            file_train = file + '.train'+str(i)+'.csv'
            file_test = file + '.test' + str(i) + '.csv'

            data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))
            data_test = pd.DataFrame(pd.read_csv(file_test, sep=";"))

            X_train = data_train.values[:, :-1]
            y_train = data_train.values[:, -1]

            X_test = data_test.values[:, :-1]
            y_test = data_test.values[:, -1]

            # CART
            clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=max(math.ceil(lamb * 2 * len(y_train)), 2),
                                              min_samples_leaf=math.ceil(lamb * len(y_train)),
                                              max_leaf_nodes=math.floor(1 / (2 * lamb)),
                                              min_impurity_decrease=lamb
                                              )
            clf = clf.fit(X_train, y_train)

            nleaves_CART = (clf.tree_.node_count + 1) / 2
            trainaccu_CART = clf.score(X_train, y_train)
            testaccu_CART = clf.score(X_test, y_test)

            with open(file_CART, 'a+') as f:
                f.write(";".join([str(i), str(lamb), str(nleaves_CART), str(trainaccu_CART), str(testaccu_CART)]) + '\n')

            # OSDT
            leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT =\
                bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=False)
            _, testaccu_OSDT = predict(leaves_c, prediction_c, dic, X_test, y_test)

            with open(file_OSDT, 'a+') as f:
                f.write(";".join(
                    [str(i), str(lamb), str(nleaves_OSDT), str(trainaccu_OSDT), str(testaccu_OSDT),
                     str(totaltime), str(time_c), str(leaves_c)]) + '\n')

"""
lambs = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001, 0.00005]

test_accuracy('../data/preprocessed/compas-binary.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_compas.txt', file_OSDT=r'./accuracy/osdt_compas.txt', timelimit=3600)  # set time limit to be 1h

test_accuracy('../data/preprocessed/balance-scale.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_balance.txt', file_OSDT=r'./accuracy/osdt_balance.txt')

test_accuracy('../data/preprocessed/car-evaluation.csv', lambs=lambs, #lambs,
              file_CART=r'./accuracy/cart_car.txt', file_OSDT=r'./accuracy/osdt_car.txt')

test_accuracy('../data/preprocessed/tic-tac-toe.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_tictactoe.txt', file_OSDT=r'./accuracy/osdt_tictactoe.txt')

test_accuracy('../data/preprocessed/monk1-train.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_monk1.txt', file_OSDT=r'./accuracy/osdt_monk1.txt')

test_accuracy('../data/preprocessed/monk2-train.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_monk2.txt', file_OSDT=r'./accuracy/osdt_monk2.txt')

test_accuracy('../data/preprocessed/monk3-train.csv', lambs=lambs,
              file_CART=r'./accuracy/cart_monk3.txt', file_OSDT=r'./accuracy/osdt_monk3.txt')

test_accuracy('../data/preprocessed/fico_binary.csv', lambs=[0.001],  #lambs,
              file_CART=r'./accuracy/cart_fico.txt', file_OSDT=r'./accuracy/osdt_fico.txt')
"""

def test_accuracy_onefold(file, lambs, file_CART, file_OSDT, timelimit):
    """
    Run CART and OSDT
    use all data, only training accuracy
    :param X:
    :param y:
    :param lambs:
    :param file_CART:
    :param file_OSDT:
    :return:
    """
    with open(file_CART, 'a+') as f:
        f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_CART", "testaccu_CART"]) + '\n')
    with open(file_OSDT, 'a+') as f:
        f.write(";".join(["fold", "lamb", "nleaves", "trainaccu_CART", "testaccu_CART", "totaltime", "time_c", "leaves_c"]) + '\n')
    for lamb in lambs:

        file_train = file

        data_train = pd.DataFrame(pd.read_csv(file_train, sep=";"))

        X_train = data_train.values[:, :-1]
        y_train = data_train.values[:, -1]


        # CART
        clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=max(math.ceil(lamb * 2 * len(y_train)), 2),
                                          min_samples_leaf=math.ceil(lamb * len(y_train)),
                                          max_leaf_nodes=math.floor(1 / (2 * lamb)),
                                          min_impurity_decrease=lamb
                                          )
        clf = clf.fit(X_train, y_train)

        nleaves_CART = (clf.tree_.node_count + 1) / 2
        trainaccu_CART = clf.score(X_train, y_train)

        with open(file_CART, 'a+') as f:
            f.write(";".join([str('NA'), str(lamb), str(nleaves_CART), str(trainaccu_CART), str('NA')]) + '\n')

        # OSDT
        leaves_c, prediction_c, dic, nleaves_OSDT, nrule, ndata, totaltime, time_c, COUNT, C_c, trainaccu_OSDT =\
            bbound(X_train, y_train, lamb=lamb, prior_metric="curiosity", timelimit=timelimit, init_cart=True)

        with open(file_OSDT, 'a+') as f:
            f.write(";".join(
                [str('NA'), str(lamb), str(nleaves_OSDT), str(trainaccu_OSDT), str('NA'),
                 str(totaltime), str(time_c), str(leaves_c)]) + '\n')

        if nleaves_OSDT >= 16:
            break



lambs1 = [0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001]

timelimi1 = 1800
timelimi2 = 7200 # set time limit to be 2h
'''
test_accuracy_onefold('../data/preprocessed/compas-binary.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_compas.txt', file_OSDT=r'./accuracy/osdt_compas.txt', timelimit=timelimi1)


test_accuracy_onefold('../data/preprocessed/car-evaluation.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_car.txt', file_OSDT=r'./accuracy/osdt_car.txt', timelimit=timelimi1)


test_accuracy_onefold('../data/preprocessed/tic-tac-toe.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_tictactoe.txt', file_OSDT=r'./accuracy/osdt_tictactoe.txt', timelimit=timelimi1)
'''
test_accuracy_onefold('../data/preprocessed/fico_binary.csv', lambs=[0.0003875], #0.05, 0.005, 0.0035, 0.0025, 0.001, 0.00075, 0.0005, 0.0001],
                      file_CART=r'./accuracy/cart_fico.txt', file_OSDT=r'./accuracy/osdt_fico.txt', timelimit=timelimi1)
'''
test_accuracy_onefold('../data/preprocessed/monk1-train.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_monk1.txt', file_OSDT=r'./accuracy/osdt_monk1.txt', timelimit=timelimi1)

test_accuracy_onefold('../data/preprocessed/monk2-train.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_monk2.txt', file_OSDT=r'./accuracy/osdt_monk2.txt', timelimit=timelimi1)


test_accuracy_onefold('../data/preprocessed/monk3-train.csv', lambs=lambs1,
                      file_CART=r'./accuracy/cart_monk3.txt', file_OSDT=r'./accuracy/osdt_monk3.txt', timelimit=timelimi1)
'''