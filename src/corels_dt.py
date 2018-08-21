import numpy as np
import heapq
import math
import time

from rule import make_all_ones, make_zeros, rule_vand, rule_vectompz

class CacheTree:
    """
    A tree data structure.
    leaves: a 2-d tuple to encode the leaves
    num_captured: a list to record number of data captured by the leaves
    """

    def __init__(self, ndata, leaves,
                 prior_metric=None,
                 splitleaf=None,
                 lbound=None
                 ):
        self.leaves = leaves
        # a queue of lists indicating which leaves will be split in next rounds
        # (1 for split, 0 for not split)
        self.splitleaf = splitleaf
        self.lbound = lbound  # a list of lower bound

        l = len(leaves)

        self.risk = self.lbound[0] + (leaves[0].p * leaves[0].num_captured) / ndata

        # which metrics to use for the priority queue
        if leaves[0].num_captured == ndata:
            # this case is when constructing the null tree ((),)
            self.metric = 0
        elif prior_metric == "curiosity":
            self.metric = min([self.lbound[i] / ((ndata - leaves[i].num_captured) / ndata)
                               if leaves[i].is_dead == 0 else float('Inf') for i in range(l)])
        elif prior_metric == "bound":
            self.metric = min([self.lbound[i] if leaves[i].is_dead == 0 else float('Inf') for i in range(l)])
        elif prior_metric == "entropy":
            # entropy weighted by number of points captured
            self.entropy = [(-leaves[i].p * math.log2(leaves[i].p) - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)) * leaves[i].num_captured
                            if leaves[i].p != 0 and leaves[i].p != 1 else 0 for i in range(l)]
            self.metric = min([sum(self.entropy[:i] + self.entropy[i + 1:]) / (ndata - leaves[i].num_captured)
                               if leaves[i].is_dead == 0 else float('Inf') for i in range(l)])
        elif prior_metric == "gini":
            # gini index weighted by number of points captured
            self.giniindex = [(2 * leaves[i].p * (1 - leaves[i].p))
                              * leaves[i].num_captured for i in range(l)]
            self.metric = min([sum(self.giniindex[:i] + self.giniindex[i + 1:]) / (ndata - leaves[i].num_captured)
                               if leaves[i].is_dead == 0 else float('Inf') for i in range(l)])
        elif prior_metric == "objective":
            self.metric = self.risk

    def sorted_leaves(self):
        # Used by the cache
        return tuple(sorted(leaf.rules for leaf in self.leaves))

    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric
"""
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

            current_node['label'] = self.leaves[i].prediction

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
"""

class CacheLeaf:
    """
    A data structure to cache every single leaf (symmetry aware)
    """

    def __init__(self, rules, y, z, points_cap, num_captured, lamb, support):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured

        # the y's of these data captured by leaf antecedent[0]
        #y_leaf = y[tag]
        # print("tag",tag)
        # print("y",y)
        _, num_ones = rule_vand(points_cap, rule_vectompz(y))

        #b0 is defined in (28)

        tag_z = rule_vectompz(z.reshape(1, -1)[0])
        _, num_errors = rule_vand(points_cap, tag_z)
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

        # Lower bound on antecedent support
        if support == True:
            self.is_dead = self.num_captured / len(y) / 2 < lamb
        else:
            self.is_dead = 0

        self.loss = float(self.num_captured_incorrect) / len(y)



def log(lines, COUNT_POP, COUNT, queue_size, metric, R_c, tree_new):
    "log"

    the_count_pop = str(COUNT_POP)
    the_count = str(COUNT)
    the_queue_size = str(queue_size)
    the_metric = str(metric)
    the_Rc = str(R_c)
    the_new_tree_objective = str(tree_new.risk)
    the_new_tree_lbound = str(min(tree_new.lbound))
    the_new_tree_length = str(len(tree_new.leaves))
    the_new_tree_depth = str(max([len(leaf.rules) for leaf in tree_new.leaves]))

    line = ";".join([the_count_pop, the_count, the_queue_size, the_metric, the_Rc,
                     the_new_tree_objective, the_new_tree_lbound, the_new_tree_length, the_new_tree_depth])
    lines.append(line)


def generate_new_splitleaf(splitleaf_list, cap_l, incorr_l, ndata, nleaves, lb, b0, lamb, R_c, accu_support, equiv_points, lookahead):
    """
    generate the new splitleaf for the new tree
    """
    splitleaf_array = np.array(splitleaf_list)
    sl = splitleaf_list.copy()

    #(Lower bound on accurate antecedent support)
    if accu_support==True:
        a_l = (sum(cap_l) - sum(incorr_l)) / ndata - sum(cap_l) / ndata / 2
    else:
        a_l = float('Inf')

    # binary vector indicating split or not
    splitleaf1 = [1] * nleaves  # all leaves labeled as to be split
    splitleaf2 = [0] * (nleaves - 2) + [1, 1]  # l1,l2 labeled as to be split
    splitleaf3 = [1] * (nleaves - 2) + [0, 0]  # dp labeled as to be split


    if lookahead==True:
        lambbb = lamb
    else:
        lambbb = 0

    if equiv_points==True:
        b00 = b0
    else:
        b00 = 0

    if lb + b00 + lambbb >= R_c:
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


def bbound(x, y, z, lamb, prior_metric=None, MAXDEPTH=4, niter=float('Inf'), logon=False,
           support=True, accu_support=True, equiv_points=True, lookahead=True):
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

    tic = time.time()

    lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf((), y, z, make_all_ones(ndata+1), ndata, lamb, support)
    tree0 = CacheTree(leaves=[root_leaf], ndata = ndata, prior_metric=prior_metric, splitleaf=[[1]], lbound=[lamb])
    heapq.heappush(queue, (tree0.metric, tree0))
    # queue.append(tree0)
    d_c = tree0
    R_c = tree0.risk
    R = tree0.risk
    #log(lines, lamb, tic, len(queue), tuple(), tree0, R, d_c, R_c)

    COUNT = 0  # count the total number of trees in the queue

    COUNT_POP = 0
    while queue and COUNT < niter:
        #tree = queue.pop(0)
        metric, tree = heapq.heappop(queue)

        COUNT_POP = COUNT_POP + 1

        #print([leaf.rules for leaf in tree.leaves])
        #print("curio", curio)
        leaves = tree.leaves

        # print("=======COUNT=======",COUNT)
        # print("d",d)
        # print("R",tree.lbound[0]+(tree.num_captured_incorrect[0])/len(y))

        """
        # if we have visited this tree
        if tree.sorted_leaves() in tree_cache:
            continue
        else:
            tree_cache[tree.sorted_leaves()] = True
        """

        # the leaves we are going to split
        split_next = tree.splitleaf.copy()
        spl = split_next.pop(0)

        # enumerate through all the leaves
        for i in range(len(leaves)):
            # print("d!!!",d)
            # if the leaf is dead, then continue
            if tree.leaves[i].is_dead == 1:
                continue

            # 0 for not split; 1 for split
            if spl[i] == 0:
                continue


            removed_leaf = leaves[i]
            unchanged_leaves = leaves[:i] + leaves[i+1:]

            # Restrict the depth of the tree
            if len(removed_leaf.rules) >= MAXDEPTH:
                continue

            # we are going to split leaf i, and get 2 new leaves
            # we will add the two new leaves to the end of the list
            splitleaf_list = [split_next[k][:i] + split_next[k][i + 1:] + split_next[k][i:i + 1] * 2
                              for k in range(len(split_next))]

            lb = tree.lbound[i]  # the lower bound
            b0 = tree.leaves[i].B0  # the b0 defined in (28) of the paper


            d0 = removed_leaf.rules

            # split the leaf d0 with feature j
            for j in range(1, nrule + 1):
                if j not in d0 and -j not in d0:
                    # split leaf d0 with feature j, and get 2 leaves l1 and l2
                    l1 = d0 + (-j,)
                    l2 = d0 + (j,)
                    # print("t",t)

                    pred_l = [0] * 2
                    cap_l = [0] * 2
                    incorr_l = [0] * 2
                    p_l = [0] * 2
                    B0_l = [0] * 2
                    points_l = make_zeros(2)

                    # for the two new leaves, if they have not been visited,
                    # calculate their predictions,
                    l1_sorted = tuple(sorted(l1))
                    l2_sorted = tuple(sorted(l2))


                    tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf

                    rule_index = j-1

                    if l1_sorted not in leaf_cache:
                        tag_rule1 = rule_vectompz(np.array(x[:, rule_index] == 0) * 1)
                        new_points_cap1, new_num_captured1 = rule_vand(tag, tag_rule1)
                        leaf_cache[l1_sorted] = CacheLeaf(l1_sorted, y, z, new_points_cap1, new_num_captured1, lamb, support)

                    Cache_l1 = leaf_cache[l1_sorted]
                    cap_l[0], incorr_l[
                        0] = Cache_l1.num_captured, Cache_l1.num_captured_incorrect

                    if l2_sorted not in leaf_cache:
                        tag_rule2 = rule_vectompz(np.array(x[:, rule_index] == 1) * 1)
                        new_points_cap2, new_num_captured2 = rule_vand(tag, tag_rule2)
                        leaf_cache[l2_sorted] = CacheLeaf(l2_sorted, y, z, new_points_cap2, new_num_captured2, lamb, support)

                    Cache_l2 = leaf_cache[l2_sorted]
                    cap_l[1], incorr_l[
                        1] = Cache_l2.num_captured, Cache_l2.num_captured_incorrect


                    new_leaves = [Cache_l1, Cache_l2]

                    sorted_new_tree_leaves = tuple(sorted(leaf.rules for leaf in unchanged_leaves+new_leaves))
                    if sorted_new_tree_leaves in tree_cache:
                        continue
                    else:
                        tree_cache[sorted_new_tree_leaves] = True


                    # calculate the bounds for each leaves in the new tree
                    loss_l1 = incorr_l[0] / ndata
                    loss_l2 = incorr_l[1] / ndata
                    loss_d0 = tree.leaves[i].p * tree.leaves[i].num_captured / ndata
                    delta = loss_l1 + loss_l2 - loss_d0 + lamb
                    old_lbound = tree.lbound[:i] + tree.lbound[i + 1:]
                    new_lbound = [b + delta for b in old_lbound] + \
                        [tree.lbound[i] + loss_l2 + lamb,
                            tree.lbound[i] + loss_l1 + lamb]

                    # generate the new splitleaf for the new tree
                    sl = generate_new_splitleaf(
                        splitleaf_list, cap_l, incorr_l, ndata, len(unchanged_leaves)+2, lb, b0, lamb, min(R_c, new_lbound[-1]+loss_l2),
                        accu_support, equiv_points, lookahead)
                    # print('sl',sl)

                    # construct the new tree
                    tree_new = CacheTree(ndata=ndata, leaves=unchanged_leaves+new_leaves,
                                         prior_metric=prior_metric,
                                         splitleaf=sl,
                                         lbound=new_lbound,
                                         )



                    # queue.append(tree_new)

                    heapq.heappush(queue, (tree_new.metric, tree_new))
                    COUNT = COUNT + 1
                    R = tree_new.risk
                    if R < R_c:
                        d_c = tree_new
                        R_c = R
                        C_c = COUNT
                        time_c = time.time()-tic

                    if logon==True:
                        log(lines, COUNT_POP, COUNT, len(queue), metric, R_c, tree_new)

                    if COUNT % 100000 == 0:
                        print("COUNT:", COUNT)


    header = ['#pop', '#push', 'queue_size', 'metric', 'R_c',
              'the_new_tree_objective', 'the_new_tree_lbound', 'the_new_tree_length', 'the_new_tree_depth']

    fname = "_".join([str(nrule), str(ndata), prior_metric,
                      str(lamb), str(MAXDEPTH), str(lookahead), ".txt"])
    with open(fname, 'w') as f:
        f.write('%s\n' % ";".join(header))
        f.write('\n'.join(lines))

    print(">>> log:",logon)
    print(">>> support bound:",support)
    print(">>> accurate support bound:",accu_support)
    print(">>> equiv points bound:",equiv_points)
    print(">>> lookahead bound:",lookahead)

    print("total time: ", time.time() - tic)
    print("lambda: ", lamb)
    print("leaves: ", [leaf.rules for leaf in d_c.leaves])
    #print("lbound: ", d_c.lbound)
    #print("d_c.num_captured: ", [leaf.num_captured for leaf in d_c.leaves])
    print("prediction: ", [leaf.prediction for leaf in d_c.leaves])
    print("Objective: ", R_c)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return d_c
