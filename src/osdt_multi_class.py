import numpy as np
import pandas as pd
import heapq
import math
import time

import copy

from itertools import product, compress
from gmpy2 import mpz
from rule import make_all_ones, make_zeros, rule_vand, rule_vandnot, rule_vectompz, rule_mpztovec, count_ones

import sklearn.tree
from sklearn.metrics import accuracy_score

import pickle


class CacheTree:
    """
    A tree data structure.
    leaves: a 2-d tuple to encode the leaves
    num_captured: a list to record number of data captured by the leaves
    """

    def __init__(self, lamb, leaves):
        self.leaves = leaves
        self.risk = sum([l.loss for l in leaves]) + lamb * len(leaves)

    def sorted_leaves(self):
        # Used by the cache
        return tuple(sorted(leaf.rules for leaf in self.leaves))

class Tree:
    """
        A tree data structure, based on CacheTree
        cache_tree: a CacheTree
        num_captured: a list to record number of data captured by the leaves
        """

    def __init__(self, cache_tree, ndata, lamb, splitleaf=None, prior_metric=None):
        self.cache_tree = cache_tree
        # a queue of lists indicating which leaves will be split in next rounds
        # (1 for split, 0 for not split)
        self.splitleaf = splitleaf

        leaves = cache_tree.leaves
        l = len(leaves)

        self.lb = sum([cache_tree.leaves[i].loss for i in range(l)
                       if splitleaf[i] == 0]) + lamb * l

        # which metrics to use for the priority queue
        if leaves[0].num_captured == ndata:
            # this case is when constructing the null tree ((),)
            self.metric = 0
        elif prior_metric == "objective":
            self.metric = cache_tree.risk
        elif prior_metric == "bound":
            self.metric = self.lb
        elif prior_metric == "curiosity":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            if num_cap_rm < ndata:
                self.metric = self.lb / ((ndata - num_cap_rm) / ndata)
            else:
                self.metric = self.lb / (0.01 / ndata)
        elif prior_metric == "entropy":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # entropy weighted by number of points captured
            self.entropy = [
                (-leaves[i].p * math.log2(leaves[i].p) - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)) * leaves[
                    i].num_captured if leaves[i].p != 0 and leaves[i].p != 1 else 0 for i in range(l)]
            if num_cap_rm < ndata:
                self.metric = sum(self.entropy[i] for i in range(l) if splitleaf[i] == 0) / (
                        ndata - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.entropy[i] for i in range(l) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "gini":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # gini index weighted by number of points captured
            self.giniindex = [(2 * leaves[i].p * (1 - leaves[i].p))
                              * leaves[i].num_captured for i in range(l)]
            if num_cap_rm < ndata:
                self.metric = sum(self.giniindex[i] for i in range(l) if splitleaf[i] == 0) / (
                        ndata - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.giniindex[i] for i in range(l) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "FIFO":
            self.metric = 0

    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric


class CacheLeaf:
    """
    A data structure to cache every single leaf (symmetry aware)
    """

    def __init__(self, ndata, rules, label_idx, y_mpz, z_mpz, points_cap, num_captured, lamb, support, is_feature_dead):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured
        self.is_feature_dead = is_feature_dead

        _, num_errors = rule_vand(points_cap, z_mpz)
        self.B0 = num_errors / ndata

        if self.num_captured:
            y_captured = np.zeros(len(label_idx))
            for i in label_idx:
                _, num_ones = rule_vand(points_cap, y_mpz[i])
                y_captured[i] = num_ones
            # if there are two dominating labels, choose either one
            pred_idx = np.argmax(y_captured)
            self.prediction = pred_idx + 1
            self.num_captured_incorrect = self.num_captured - np.max(y_captured)

            self.p = self.num_captured_incorrect / self.num_captured
        else:
            self.prediction = 0
            self.num_captured_incorrect = 0
            self.p = 0

        self.loss = float(self.num_captured_incorrect) / ndata

        # Lower bound on leaf support
        if support:
            self.is_dead = self.loss <= lamb
        else:
            self.is_dead = 0


def log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree_old, tree_new, sorted_new_tree_rules, fname):
    "log"

    the_time = str(time.time() - tic)

    the_count_pop = str(COUNT_POP)
    the_count = str(COUNT)
    the_queue_size = str(0)  # str(len(queue))
    the_metric = str(metric)
    the_Rc = str(R_c)

    the_old_tree = str(0)  # str(sorted([leaf.rules for leaf in tree_old.cache_tree.leaves]))
    the_old_tree_splitleaf = str(0)  # str(tree_old.splitleaf)
    the_old_tree_objective = str(tree_old.cache_tree.risk)
    the_old_tree_lbound = str(tree_old.lb)
    the_new_tree = str(0)  # str(list(sorted_new_tree_rules))
    the_new_tree_splitleaf = str(0)  # str(tree_new.splitleaf)

    the_new_tree_objective = str(0)  # str(tree_new.cache_tree.risk)
    the_new_tree_lbound = str(tree_new.lb)
    the_new_tree_length = str(0)  # str(len(tree_new.cache_tree.leaves))
    the_new_tree_depth = str(0)  # str(max([len(leaf.rules) for leaf in tree_new.leaves]))

    the_queue = str(0)  # str([[ leaf.rules for leaf in thetree.leaves]  for _,thetree in queue])

    line = ";".join([the_time, the_count_pop, the_count, the_queue_size, the_metric, the_Rc,
                     the_old_tree, the_old_tree_splitleaf, the_old_tree_objective, the_old_tree_lbound,
                     the_new_tree, the_new_tree_splitleaf,
                     the_new_tree_objective, the_new_tree_lbound, the_new_tree_length, the_new_tree_depth,
                     the_queue
                     ])

    with open(fname, 'a+') as f:
        f.write(line+'\n')


def generate_new_splitleaf(unchanged_leaves, removed_leaves, new_leaves, lamb,
                           R_c, incre_support):
    """
    generate the new leaf to split for the new tree
    """

    n_removed_leaves = len(removed_leaves)
    n_unchanged_leaves = len(unchanged_leaves)
    n_new_leaves = len(new_leaves)

    n_new_tree_leaves = n_unchanged_leaves + n_new_leaves

    splitleaf1 = [0] * n_unchanged_leaves + [1] * n_new_leaves  # all new leaves labeled as to be split
    splitleaf1 = tuple(splitleaf1)
    sl = []
    for i in range(n_removed_leaves):

        splitleaf = [0] * n_new_tree_leaves

        idx1 = 2*i
        idx2 = 2*i+1
        # (Lower bound on incremental classification accuracy)

        a_l = removed_leaves[i].loss - new_leaves[idx1].loss - new_leaves[idx2].loss

        if not incre_support:
            a_l = float('Inf')

        if a_l <= lamb:
            splitleaf[n_unchanged_leaves + idx1] = 1
            splitleaf[n_unchanged_leaves + idx2] = 1
            splitleaf = tuple(splitleaf)
            sl.append(splitleaf)
        else:
            sl.append(splitleaf1)
    # remove duplicates
    sl = list(set(sl))
    return sl




def gini_reduction(x_mpz, y_mpz, ndata, rule_idx, lable_idx, points_cap=None):
    """
    calculate the gini reduction by each feature
    return the rank of by descending
    """

    if points_cap == None:
        points_cap = make_all_ones(ndata + 1)

    gini0 = 1
    for i in lable_idx:
        yi = y_mpz[i]
        _, ndata1 = rule_vand(yi, points_cap)
        p = ndata1/ndata
        gini0 -= p**2


    gr = []
    for i in rule_idx:
        xi = x_mpz[i]
        l1_cap, ndata1 = rule_vand(points_cap, ~xi | mpz(pow(2, ndata))) # ndata1 = number of data that xi = 0

        gini1 = 1
        for j in lable_idx:
            yj = y_mpz[j]
            _, ndata11 = rule_vand(l1_cap, yj)
            p1 = ndata11/ndata1 if ndata1 != 0 else 0
            gini1 -= p1**2

        l2_cap, ndata2 = rule_vand(points_cap, xi) # ndata2 = number of data that xi = 1

        gini2 = 1
        for j in lable_idx:
            yj = y_mpz[j]
            _, ndata21 = rule_vand(l2_cap, yj)
            p2 = ndata21 / ndata2 if ndata2 != 0 else 0
            gini1 -= p2 ** 2

        gini_red = gini0 - ndata1 / ndata * gini1 - ndata2 / ndata * gini2
        gr.append(gini_red)

    gr = np.array(gr)
    order = list(gr.argsort()[::-1])
    odr = [rule_idx[r] for r in order]
    dic = dict(zip(np.array(rule_idx)+1, odr))

    return odr, dic


def get_code(tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for scikit-leant DescisionTree.

        Args
        ----
        tree -- scikit-leant DescisionTree.
        feature_names -- list of feature names.
        target_names -- list of target (class) names.
        spacer_base -- used for spacing code (default: "    ").

        Notes
        -----
        based on http://stackoverflow.com/a/30104792.
        http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
        """
    #tree  # = dt
    #feature_names   #= features
    #target_names   #= targets

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    feats = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print((spacer + "if ( " + feats[node] + " <= " + str(threshold[node]) + " ) {"))
            if left[node] != -1:
                recurse(left, right, threshold, feats, left[node], depth + 1)
            print((spacer + "}\n" + spacer + "else {"))
            if right[node] != -1:
                recurse(left, right, threshold, feats, right[node], depth + 1)
            print((spacer + "}"))
        else:
            target = value[node]
            print((spacer + "return " + str(target)))
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print((spacer + "return " + str(target_name) + " " + str(i) + " " \
                                                                              " ( " + str(
                    target_count) + " examples )"))

    recurse(left, right, threshold, feature_names, 0, 0)

def bbound(x, y, lamb, prior_metric=None, MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=True, R_c0 = 1, timelimit=float('Inf'), init_cart = True,
           saveTree = False, readTree = False):
    """
    An implementation of Algorithm
    ## multiple copies of tree
    ## mark which leaves to be split
    """

    x0 = copy.deepcopy(x)
    y0 = copy.deepcopy(y)

    # start the timer
    tic = time.time()

    nrule = x.shape[1]
    ndata = x.shape[0]
    nlabel = y.shape[1]

    max_nleaves = 2**nrule
    print("nrule:", nrule)
    print("ndata:", ndata)
    print("nlabel:", nlabel)

    x_mpz = [rule_vectompz(x[:, i]) for i in range(nrule)]
    y_mpz = [rule_vectompz(y[:, i]) for i in range(nlabel)]

    # order the columns by descending gini reduction
    idx, dic = gini_reduction(x_mpz, y_mpz, ndata, range(nrule), range(nlabel))
    x = x[:, idx]
    x_mpz = [x_mpz[i] for i in idx]
    print("the order of x's columns: ", idx)

    """
    calculate z, which is for the equivalent points bound
    z is the vector defined in algorithm 5 of the CORELS paper
    z is a binary vector indicating the data with a minority label in its equivalent set
    """

    z = pd.DataFrame([-1]*ndata).values
    # enumerate through theses samples
    for i in range(ndata):
        # if z[i,0]==-1, this sample i has not been put into its equivalent set
        if z[i, 0] == -1:
            tag1 = np.array([True] * ndata)
            for j in range(nrule):
                rule_label = x[i][j]
                # tag1 indicates which samples have exactly the same features with sample i
                tag1 = (x[:, j] == rule_label) * tag1

            y_l = y[tag1]
            yl = np.sum(y_l,axis=0)
            idx_common = np.where(yl == np.amax(yl))  # only one majority label for now
            idx_common = np.array(idx_common)
            pred = np.zeros((np.size(idx_common), nlabel))
            for i in range(idx_common.shape[1]):
                pred[i, idx_common[0, i]] = 1
            tag2 = np.full((y_l.shape[0]), False)
            for i in range(pred.shape[0]):
                tag2 = tag2 | (y_l == pred[i]).all(1)
            # tag2 indicates the samples in a equiv set which have the minority label
            tag2 = ~tag2
            z[tag1, 0] = tag2

    z_mpz = rule_vectompz(z.reshape(1, -1)[0])


    lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf(ndata, (), range(nlabel), y_mpz, z_mpz, make_all_ones(ndata + 1), ndata, lamb, support, [0] * nrule)

    d_c = CacheTree(leaves=[root_leaf], lamb=lamb)
    R_c = d_c.risk

    tree0 = Tree(cache_tree=d_c, lamb=lamb,
                 ndata=ndata, splitleaf=[1], prior_metric=prior_metric)

    heapq.heappush(queue, (tree0.metric, tree0))


    # read Tree from the preserved one, and only explore the children of the preserved one
    if readTree:
        with open('tree.pkl', 'rb') as f:
            d_c = pickle.load(f)
        R_c = d_c.risk

        with open('leaf_cache.pkl', 'rb') as f:
            leaf_cache = pickle.load(f)

        sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in d_c.leaves))
        tree_cache[sorted_new_tree_rules] = True

        tree_p = Tree(cache_tree=d_c, lamb=lamb,
                     ndata=ndata, splitleaf=[1]*len(d_c.leaves), prior_metric=prior_metric)

        heapq.heappush(queue, (tree_p.metric, tree_p))
        print("PICKEL>>>>>>>>>>>>>", [leaf.rules for leaf in d_c.leaves])
        #print("leaf_cache:", leaf_cache)

        C_c = 0
        time_c = time.time() - tic

    if R_c0 < R_c:
        R_c = R_c0

    # log(lines, lamb, tic, len(queue), tuple(), tree0, R, d_c, R_c)

    leaf_cache[()] = root_leaf

    COUNT = 0  # count the total number of trees in the queue

    COUNT_POP = 0

    COUNT_UNIQLEAVES = 0
    COUNT_LEAFLOOKUPS = 0


    if logon:
        header = ['time', '#pop', '#push', 'queue_size', 'metric', 'R_c',
                  'the_old_tree', 'the_old_tree_splitleaf', 'the_old_tree_objective', 'the_old_tree_lbound',
                  'the_new_tree', 'the_new_tree_splitleaf',
                  'the_new_tree_objective', 'the_new_tree_lbound', 'the_new_tree_length', 'the_new_tree_depth', 'queue']

        fname = "_".join([str(nrule), str(ndata), prior_metric,
                          str(lamb), str(MAXDEPTH), str(init_cart), ".txt"])
        with open(fname, 'w') as f:
            f.write('%s\n' % ";".join(header))

    while queue and COUNT < niter and time.time() - tic < timelimit:
        metric, tree = heapq.heappop(queue)

        COUNT_POP = COUNT_POP + 1

        leaves = tree.cache_tree.leaves
        leaf_split = tree.splitleaf
        removed_leaves = list(compress(leaves, leaf_split))
        old_tree_length = len(leaf_split)
        new_tree_length = old_tree_length + sum(leaf_split)

        # prefix-specific upper bound on number of leaves
        if lenbound and new_tree_length >= min(old_tree_length + math.floor((R_c - tree.lb) / lamb),
                                               max_nleaves):
            continue

        n_removed_leaves = sum(leaf_split)
        n_unchanged_leaves = old_tree_length - n_removed_leaves

        # equivalent points bound combined with the lookahead bound
        lb = tree.lb
        b0 = sum([leaf.B0 for leaf in removed_leaves]) if equiv_points else 0
        lambbb = lamb if lookahead else 0
        if lb + b0 + n_removed_leaves * lambbb >= R_c:
            continue

        leaf_no_split = [not split for split in leaf_split]
        unchanged_leaves = list(compress(leaves, leaf_no_split))

        # Generate all assignments of rules to the leaves that are due to be split

        rules_for_leaf = [set(range(1, nrule + 1)) - set(map(abs, l.rules)) -
                          set([i+1 for i in range(nrule) if l.is_feature_dead[i] == 1]) for l in removed_leaves]


        for leaf_rules in product(*rules_for_leaf):

            if time.time() - tic >= timelimit:
                break

            new_leaves = []
            flag_increm = False  # a flag for jump out of the loops (incremental support bound)
            for rule, removed_leaf in zip(leaf_rules, removed_leaves):

                rule_index = rule - 1
                tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf

                for new_rule in (-rule, rule):  # 0 or 1 feature label
                    new_rule_label = int(new_rule > 0)
                    new_rules = tuple(
                        sorted(removed_leaf.rules + (new_rule,)))
                    if new_rules not in leaf_cache:

                        COUNT_UNIQLEAVES = COUNT_UNIQLEAVES+1

                        tag_rule = x_mpz[rule_index] if new_rule_label == 1 else ~(x_mpz[rule_index]) | mpz(pow(2, ndata))

                        new_points_cap, new_num_captured = rule_vand(tag, tag_rule)
                        # generate new leaves
                        new_leaf = CacheLeaf(ndata, new_rules, range(nlabel), y_mpz, z_mpz, new_points_cap,
                                             new_num_captured, lamb, support, removed_leaf.is_feature_dead.copy())
                        leaf_cache[new_rules] = new_leaf
                        new_leaves.append(new_leaf)
                    else:

                        COUNT_LEAFLOOKUPS = COUNT_LEAFLOOKUPS+1

                        new_leaf = leaf_cache[new_rules]
                        new_leaves.append(new_leaf)

                    if accu_support and (new_leaf.num_captured - new_leaf.num_captured_incorrect) / ndata <= lamb:

                        removed_leaf.is_feature_dead[rule_index] = 1

                        flag_increm = True
                        break

                if flag_increm:
                    break

            if flag_increm:
                continue

            new_tree_leaves = unchanged_leaves + new_leaves

            sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in new_tree_leaves))

            if sorted_new_tree_rules in tree_cache:  # have explored the same subtree
                continue
            else:
                tree_cache[sorted_new_tree_rules] = True

            child = CacheTree(leaves=new_tree_leaves, lamb=lamb)

            R = child.risk
            if R < R_c:
                # current optimal tree
                d_c = child
                R_c = R
                C_c = COUNT + 1
                time_c = time.time() - tic

            # generate the new splitleaf for the new tree
            sl = generate_new_splitleaf(unchanged_leaves, removed_leaves, new_leaves,
                                        lamb, R_c, incre_support)

            # A leaf cannot be split if
            # 1. the MAXDEPTH has been reached
            # 2. the leaf is dead (because of antecedent support)
            # 3. all the features that have not been used are dead
            cannot_split = [len(l.rules) >= MAXDEPTH or l.is_dead or
                            all([l.is_feature_dead[r - 1] for r in range(1, nrule + 1)
                                 if r not in map(abs, l.rules)])
                            for l in new_tree_leaves]

            # For each copy, we don't split leaves which are not split in its parent tree.
            # In this way, we can avoid duplications.
            can_split_leaf = [(0,)] * n_unchanged_leaves + \
                             [(0,) if cannot_split[i]
                              else (0, 1) for i in range(n_unchanged_leaves, new_tree_length)]
            # Discard the first element of leaf_splits, since we must split at least one leaf
            new_leaf_splits0 = np.array(list(product(*can_split_leaf))[1:])
            len_sl = len(sl)
            if len_sl == 1:
                # Filter out those which split at least one leaf in dp (d0)
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if np.dot(ls, sl[0]) > 0]
            else:
                # Filter out those which split at least one leaf in dp and split at least one leaf in d0
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if all([np.dot(ls, sl[i]) > 0 for i in range(len_sl)])]

            for new_leaf_split in new_leaf_splits:
                # construct the new tree
                tree_new = Tree(cache_tree=child, ndata=ndata, lamb=lamb,
                                splitleaf=new_leaf_split, prior_metric=prior_metric)

                # MAX Number of leaves
                if len(new_leaf_split)+sum(new_leaf_split) > MAX_NLEAVES:
                    continue
                # tree counter
                COUNT = COUNT + 1
                heapq.heappush(queue, (tree_new.metric, tree_new))

                if logon:
                    log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree, tree_new, sorted_new_tree_rules, fname)

                if COUNT % 1000000 == 0:
                    print("COUNT:", COUNT)

    totaltime = time.time() - tic

    accu = 1-(R_c-lamb*len(d_c.leaves))

    leaves_c = [leaf.rules for leaf in d_c.leaves]
    prediction_c = [leaf.prediction for leaf in d_c.leaves]

    num_captured = [leaf.num_captured for leaf in d_c.leaves]

    num_captured_incorrect = [leaf.num_captured_incorrect for leaf in d_c.leaves]

    nleaves = len(leaves_c)

    if saveTree:
        with open('tree.pkl', 'wb') as f:
            pickle.dump(d_c, f)
        with open('leaf_cache.pkl', 'wb') as f:
            pickle.dump(leaf_cache, f)


    print(">>> log:", logon)
    print(">>> support bound:", support)
    print(">>> accu_support:", accu_support)
    print(">>> accurate support bound:", incre_support)
    print(">>> equiv points bound:", equiv_points)
    print(">>> lookahead bound:", lookahead)
    print("prior_metric=", prior_metric)

    print("COUNT_UNIQLEAVES:", COUNT_UNIQLEAVES)
    print("COUNT_LEAFLOOKUPS:", COUNT_LEAFLOOKUPS)

    print("total time: ", totaltime)
    print("lambda: ", lamb)
    print("leaves: ", leaves_c)
    print("num_captured: ", num_captured)
    print("num_captured_incorrect: ", num_captured_incorrect)
    print("prediction: ", prediction_c)
    print("Objective: ", R_c)
    print("Accuracy: ", accu)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return leaves_c, prediction_c, dic, nleaves, nrule, ndata, totaltime, time_c, COUNT, C_c, accu


def predict(leaves_c, prediction_c, dic, x, y):
    """

    :param leaves_c:
    :param dic:
    :return:
    """

    ndata = x.shape[0]

    caps = []

    for leaf in leaves_c:
        cap = np.array([1] * ndata)
        for feature in leaf:
            idx = dic[abs(feature)]
            feature_label = int(feature > 0)
            cap = (x[:, idx] == feature_label) * cap
        caps.append(cap)

    yhat = np.array([0] * ndata)

    for j in range(len(caps)):
        idx_cap = [i for i in range(ndata) if caps[j][i] == 1]
        yhat[idx_cap] = prediction_c[j]

    right = yhat==y
    accu = right.mean()

    print("Testing Accuracy:", accu)

    return yhat, accu