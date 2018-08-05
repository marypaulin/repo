import numpy as np
import heapq
import math
import time
from gmpy2 import mpz  # TODO: remove this, only use rule.py

from rule import make_all_ones, rule_vand, rule_vectompz


class CacheTree:
    """
    A tree data structure.
    prefix: a 2-d tuple to encode the leaves
    num_captured: a list to record number of data captured by the leaves
    """

    def __init__(self, x, y, prefix,
                 lamb, prior_metric=None,
                 num_captured=None,
                 deadleaf=None,
                 splitleaf=None,
                 lbound=None,
                 p=None,
                 B0=None, points_cap=None, prediction=None):
        self.prefix = prefix
        self.num_captured = num_captured
        #self.num_captured_incorrect = num_captured_incorrect
        self.p = p  # the proportion of misclassified data in each leaf
        # a list indicate which leaves will never be split (support bound)
        self.deadleaf = deadleaf
        # a queue of lists indicating which leaves will be split in next rounds
        # (1 for split, 0 for not split)
        self.splitleaf = splitleaf
        self.lbound = lbound  # a list of lower bound
        self.B0 = B0  # a list of b0
        # a list of mpz, indicating which data are captured by each leaf
        self.points_cap = points_cap
        self.prediction = prediction  # a list of predictions of every leaf

        ndata = len(y)
        l = len(prefix)

        self.risk = self.lbound[0] + (self.p[0] * self.num_captured[0]) / ndata

        # print(self.prefix)
        # print(self.lbound)
        # print(self.splitleaf)
        # which metrics to use for the priority queue
        if self.num_captured[0] == ndata:
            # this case is when constructing the null tree ((),)
            self.metric = 0
        elif prior_metric == "curiosity":
            self.metric = min([self.lbound[i] / ((ndata - self.num_captured[i]) / len(y))
                               if self.splitleaf[0][i] == 1 else float('Inf') for i in range(l)])
        elif prior_metric == "bound":
            self.metric = min([self.lbound[i] if self.splitleaf[0][
                              i] == 1 else float('Inf') for i in range(l)])
        elif prior_metric == "entropy":
            # entropy weighted by number of points captured
            self.entropy = [(-self.p[i] * math.log2(self.p[i]) - (1 - self.p[i]) * math.log2(1 - self.p[i])) * self.num_captured[i]
                            if self.p[i] != 0 and self.p[i] != 1 else 0 for i in range(l)]
            self.metric = min([sum(self.entropy[:i] + self.entropy[i + 1:]) / (ndata - self.num_captured[i])
                               if ndata - self.num_captured[i] != 0 else 0 for i in range(l)])
        elif prior_metric == "gini":
            # gini index weighted by number of points captured
            self.giniindex = [(2 * self.p[i] * (1 - self.p[i]))
                              * self.num_captured[i] for i in range(l)]
            self.metric = min([sum(self.giniindex[:i] + self.giniindex[i + 1:]) / (ndata - self.num_captured[i])
                               if ndata - self.num_captured[i] != 0 else 0 for i in range(l)])
        else:
            self.metric = 0

    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric

    def _to_nested_dict(self):
        tree = {}

        for i, leaf in enumerate(self.leaves):
            current_node = tree

            for rule in leaf:
                current_node['rule'] = abs(rule)
                direction = 'left' if rule < 0 else 'right'
                if direction not in current_node:
                    current_node[direction] = {}
                current_node = current_node[direction]

            current_node['label'] = self.labels[i]

        return tree

    def _format_dict(self, tree, depth=0):
        fmt = '-' * depth

        if 'rule' in tree:
            fmt += 'r{}'.format(tree['rule'])
        else:
            assert 'label' in tree
            fmt += str(tree['label'])

        fmt += '\n'

        if 'left' in tree:
            fmt += self._format_dict(tree['left'], depth + 1)

        if 'right' in tree:
            fmt += self._format_dict(tree['right'], depth + 1)

        return fmt

    def __str__(self):
        return self._format_dict(self._to_nested_dict())


class CacheLeaf:
    """
    A data structure to cache every single leaf (symmetry aware)
    """

    def __init__(self, antecedent, x, y, z, parent_points_cap):
        tag = parent_points_cap  # points captured by the leaf's parent leaf
        rule_index = abs(antecedent[0][-1]) - 1  # the leaf's last feature
        # this binary feature is 0 or 1
        rule_label = int(antecedent[0][-1] > 0)
        #print("np.array(x[:,rule_index] == rule_label)",x[:,rule_index] == rule_label)
        tag_rule = rule_vectompz(np.array(x[:, rule_index] == rule_label) * 1)
        tag, self.num_captured = rule_vand(tag, tag_rule)

        self.points_cap = tag

        # the y's of these data captured by leaf antecedent[0]
        #y_leaf = y[tag]
        # print("tag",tag)
        # print("y",y)
        _, num_ones = rule_vand(tag, rule_vectompz(y))

        #b0 is defined in (28)

        tag_z = rule_vectompz(z.reshape(1, -1)[0])
        _, num_errors = rule_vand(tag, tag_z)
        self.B0 = num_errors / len(y)

        if self.num_captured:
            self.prediction = int(num_ones / self.num_captured >= 0.5)
            if self.prediction == 1:
                self.num_captured_incorrect = self.num_captured - num_ones
            else:
                self.num_captured_incorrect = num_ones
            self.p = self.num_captured_incorrect / self.num_captured
        else:
            self.prediction = 0
            self.num_captured_incorrect = 0
            self.p = 0


class Eliminate:
    """
    A data structure to record and identify
    whether a tree has been visited/pruned
    """

    def __init__(self, elim_dict=None):
        self.elim_dict = {}  # record these trees we have visited

    def eliminate(self, prefix):
        self.elim_dict[tuple(sorted(prefix))] = 1

    def is_duplicated(self, prefix):
        # if a tree is in the self.elim_dict, then we have already visited it
        return tuple(sorted(prefix)) in self.elim_dict.keys()


def log(lines, lamb, tic, queue_size, prefix_old, tree_new, R, d_c, R_c):
    "log"
    t = tree_new.prefix
    t_c = d_c.prefix

    the_time = str(time.time() - tic)
    the_queue_size = str(queue_size)
    the_split_tree = str(prefix_old)
    the_new_tree = str(t)
    the_new_tree_prediction = str(tree_new.prediction)
    the_new_tree_length = str(len(t))
    the_new_tree_objective = str(R)
    the_new_tree_accuracy = str(1 - (R - lamb * len(tree_new.prefix)))
    the_best_tree = str(t_c)
    the_best_tree_prediction = str(d_c.prediction)
    the_length = str(len(t_c))
    the_obj = str(R_c)
    the_lbound = str(d_c.lbound)
    the_best_tree_accuracy = str(1 - (R_c - lamb * len(t_c)))
    the_num_cap = str(d_c.num_captured)

    line = ";".join([the_time, the_queue_size, the_split_tree,
                     the_new_tree, the_new_tree_prediction,
                     the_new_tree_length, the_new_tree_objective, the_new_tree_accuracy,
                     the_best_tree, the_best_tree_prediction, the_length, the_obj,
                     the_lbound, the_best_tree_accuracy, the_num_cap])
    lines.append(line)


def generate_new_splitleaf(splitleaf_list, cap_l, incorr_l, ndata, t, lb, b0, lamb, R_c):
    """
    generate the new splitleaf for the new tree
    """
    splitleaf_array = np.array(splitleaf_list)
    sl = splitleaf_list.copy()

    #(Lower bound on accurate antecedent support)
    a_l = (sum(cap_l) - sum(incorr_l)) / ndata - sum(cap_l) / ndata / 2

    # binary vector indicating split or not
    splitleaf1 = [1] * (len(t))  # all leaves labeled as to be split
    splitleaf2 = [0] * (len(t) - 2) + [1, 1]  # l1,l2 labeled as to be split
    splitleaf3 = [1] * (len(t) - 2) + [0, 0]  # dp labeled as to be split

    if lb + b0 + lamb >= R_c or lb >= R_c:
        # print("lb+b0+lamb",lb+b0+lamb)
        # print("R_c",R_c)
        # if equivalent points bound combined with the lookahead bound doesn't hold
        # or if the hierarchical objective lower bound doesn't hold
        # we need to split at least one leaf in dp

        if a_l < lamb:
            # if the bound doesn't hold, we need to split the leaf l1/l2
            # further

            if len(splitleaf_list) > 0:
                split_l1_l2 = splitleaf_array[
                    :, -1].sum() + splitleaf_array[:, -2].sum()

                # if dp will have been split
                if splitleaf_array.sum() - split_l1_l2 > 0:

                    # if l1/l2 will have been split
                    if split_l1_l2 > 0:
                        sl.append(splitleaf1)

                    # if l1/l2 will not have been split, we need to split l1/l2
                    else:
                        sl.append(splitleaf2)

                # and we need to split leaves in dp, if dp will not have been
                # split
                else:

                    # if l1/l2 will have been split
                    if split_l1_l2 > 0:
                        sl.append(splitleaf3)

                    # if l1/l2 will not have been split, we need to split l1/l2
                    else:
                        sl.append(splitleaf2)
                        sl.append(splitleaf3)
            else:
                sl.append(splitleaf2)
                sl.append(splitleaf3)

        else:

            if len(splitleaf_list) > 0:
                split_l1_l2 = splitleaf_array[
                    :, -1].sum() + splitleaf_array[:, -2].sum()

                # if dp will have been split
                if splitleaf_array.sum() - split_l1_l2 > 0:
                    sl.append(splitleaf1)

                # and we need to split leaves in dp, if dp will not have been
                # split
                else:
                    sl.append(splitleaf3)
            else:
                sl.append(splitleaf3)
    else:

        if a_l < lamb:
            # if the bound doesn't hold, we need to split the leaf l1/l2
            # further

            if len(splitleaf_list) > 0:
                split_l1_l2 = splitleaf_array[
                    :, -1].sum() + splitleaf_array[:, -2].sum()

                # if l1/l2 will have been split
                if split_l1_l2 > 0:
                    sl.append(splitleaf1)

                # if l1/l2 will not have been split, we need to split l1/l2
                else:
                    sl.append(splitleaf2)
            else:
                sl.append(splitleaf2)

        else:
            sl.append(splitleaf1)

    return sl


def bbound(x, y, z, lamb, prior_metric=None, MAXDEPTH=4, niter=float('Inf')):
    """
    An implementation of Algorithm
    ## one copy of tree
    ## mark which leaves to be split
    """

    # Initialize best rule list and objective
    #d_c = None
    #R_c = 1

    nrule = x.shape[1]
    ndata = len(y)
    print("nrule:", nrule)
    print("ndata:", ndata)

    E = Eliminate()
    tic = time.time()

    lines = []  # a list for log
    leaves = {}  # cache leaves

    # initialize the queue to include just empty root
    queue = []
    t = ((),)
    tree0 = CacheTree(prefix=t, x=x, y=y, lamb=lamb, prior_metric=prior_metric,
                      num_captured=[ndata], deadleaf=[0], splitleaf=[[1]], lbound=[lamb],
                      p=[min(np.mean(y), 1 - np.mean(y))], B0=[np.sum(z) / ndata],
                      points_cap=[make_all_ones(ndata + 1)], prediction=[int(sum(y) / len(y) >= 0.5)])
    heapq.heappush(queue, (tree0.metric, tree0))
    # queue.append(tree0)
    d_c = tree0
    R_c = tree0.risk
    R = tree0.risk
    log(lines, lamb, tic, len(queue), tuple(), tree0, R, d_c, R_c)

    COUNT = 0  # count the total number of trees in the queue
    while (queue) and COUNT < niter:
        #tree = queue.pop(0)
        (curio, tree) = heapq.heappop(queue)
        d = tree.prefix

        # print("=======COUNT=======",COUNT)
        # print("d",d)
        # print("R",tree.lbound[0]+(tree.num_captured_incorrect[0])/len(y))

        # if we have visited this tree
        if E.is_duplicated(d):
            continue
        else:
            E.eliminate(d)

        # the leaves we are going to split
        split_next = tree.splitleaf.copy()
        spl = split_next.pop(0)

        # enumerate through all the leaves
        for i in range(len(d)):
            # print("d!!!",d)
            # if the leaf is dead, then continue
            if tree.deadleaf[i] == 1:
                continue

            #(Lower bound on antecedent support)
            # if this bound doesnot hold, set the leaf to be dead, and continue
            if tree.num_captured[i] / ndata / 2 < lamb:
                tree.deadleaf[i] = 1
                continue

            # 0 for not split; 1 for split
            # if spl[0][i]==0:
            if spl[i] == 0:
                continue

            d0 = d[i]  # d0 is the leaf we are going to split
            dp = d[:i] + d[i + 1:]  # dp is the rest

            # Restrict the depth of the tree
            if len(d0) >= MAXDEPTH:
                continue

            # we are going to split leaf i, and get 2 new leaves
            # we will add the two new leaves to the end of the list
            splitleaf_list = [split_next[k][:i] + split_next[k][i + 1:] + split_next[k][i:i + 1] * 2
                              for k in range(len(split_next))]

            lb = tree.lbound[i]  # the lower bound
            # print("tree.B0",tree.B0)
            b0 = tree.B0[i]  # the b0 defined in (28) of the paper

            # split the leaf d0 with feature j
            for j in range(1, nrule + 1):
                if j not in d0 and -j not in d0:
                    # split leaf d0 with feature j, and get 2 leaves l1 and l2
                    l1 = d0 + (-j,)
                    l2 = d0 + (j,)
                    t = dp + (l1, l2)  # t is the new tree
                    # print("t",t)

                    # if tree t is duplicated, continue
                    if E.is_duplicated(t):
                        continue

                    pred_l = [0] * 2
                    cap_l = [0] * 2
                    incorr_l = [0] * 2
                    p_l = [0] * 2
                    B0_l = [0] * 2
                    points_l = [mpz(0)] * 2

                    # for the two new leaves, if they have not been visited,
                    # calculate their predictions,
                    l1_sorted = tuple(sorted(l1))
                    l2_sorted = tuple(sorted(l2))

                    i_points = tree.points_cap[i]

                    if l1_sorted not in leaves:
                        leaves[l1_sorted] = CacheLeaf((l1,), x, y, z, i_points)

                    Cache_l1 = leaves[l1_sorted]
                    cap_l[0], incorr_l[
                        0] = Cache_l1.num_captured, Cache_l1.num_captured_incorrect,

                    if l2_sorted not in leaves:
                        leaves[l2_sorted] = CacheLeaf((l2,), x, y, z, i_points)

                    Cache_l2 = leaves[l2_sorted]
                    cap_l[1], incorr_l[
                        1] = Cache_l2.num_captured, Cache_l2.num_captured_incorrect

                    # calculate the bounds for each leaves in the new tree
                    loss_l1 = incorr_l[0] / ndata
                    loss_l2 = incorr_l[1] / ndata
                    loss_d0 = tree.p[i] * tree.num_captured[i] / ndata
                    delta = loss_l1 + loss_l2 - loss_d0 + lamb
                    old_lbound = tree.lbound[:i] + tree.lbound[i + 1:]
                    new_lbound = [b + delta for b in old_lbound] + \
                        [tree.lbound[i] + loss_l2 + lamb,
                            tree.lbound[i] + loss_l1 + lamb]

                    # generate the new splitleaf for the new tree
                    sl = generate_new_splitleaf(
                        splitleaf_list, cap_l, incorr_l, ndata, t, lb, b0, lamb, R_c)
                    #print("splitleaf_list, cap_l, incorr_l, ndata, t, lb, b0, lamb, R_c",splitleaf_list, cap_l, incorr_l, ndata, t, lb, b0, lamb, R_c)
                    # print('sl',sl)

                    # construct the new tree
                    tree_new = CacheTree(x=x, y=y, prefix=t,
                                         num_captured=tree.num_captured[
                                             :i] + tree.num_captured[i + 1:] + cap_l,
                                         #num_captured_incorrect = tree.num_captured_incorrect[:i]+tree.num_captured_incorrect[i+1:]+incorr_l,
                                         deadleaf=tree.deadleaf[
                                             :i] + tree.deadleaf[i + 1:] + [0, 0],
                                         splitleaf=sl,
                                         lbound=new_lbound,
                                         p=tree.p[:i] + tree.p[i + 1:] + \
                                         [Cache_l1.p, Cache_l2.p],
                                         B0=tree.B0[:i] + tree.B0[i + 1:] + \
                                         [Cache_l1.B0, Cache_l2.B0],
                                         lamb=lamb,
                                         prior_metric=prior_metric,
                                         points_cap=tree.points_cap[
                                             :i] + tree.points_cap[i + 1:] + [Cache_l1.points_cap, Cache_l2.points_cap],
                                         prediction=tree.prediction[
                                             :i] + tree.prediction[i + 1:] + [Cache_l1.prediction, Cache_l2.prediction]
                                         )

                    # queue.append(tree_new)
                    """print("t:",t)
                    print("tree_new.num_captured:",tree_new.num_captured)
                    print("tree_new.deadleaf:",tree_new.deadleaf)
                    print("tree_new.splitleaf:",tree_new.splitleaf)
                    print("tree_new.p:",tree_new.p)
                    print("tree_new.B0:", tree_new.B0)"""

                    heapq.heappush(queue, (tree_new.metric, tree_new))
                    COUNT = COUNT + 1
                    R = tree_new.risk
                    if R < R_c:
                        d_c = tree_new
                        R_c = R
                        C_c = COUNT

                    if COUNT % 100000 == 0:
                        print("COUNT:", COUNT)

        del tree

    header = ['time', 'queue_size', 'split_tree', 'new_tree', 'new_tree_prediction', 'new_tree_length', 'new_tree_objective', 'new_tree_accuracy',
              'best_tree', 'best_tree_prediction', 'best_tree_length', 'objective', 'lower_bound', 'accuracy', 'num_captured']

    fname = "_".join([str(nrule), str(ndata), prior_metric,
                      str(lamb), str(MAXDEPTH), ".txt"])
    with open(fname, 'w') as f:
        f.write('%s\n' % ";".join(header))
        f.write('\n'.join(lines))

    """for i in range(len(d_c.prefix)):
        lea = d_c.prefix[i]
        leaves[lea].prediction"""

    print("time: ", time.time() - tic)
    print("lambda: ", lamb)
    print("d_c: ", d_c.prefix)
    print("d_c: ", d_c.prediction)
    print("R_c: ", R_c)
    print("C_c: ", C_c)
    print("COUNT: ", COUNT)

    return d_c
