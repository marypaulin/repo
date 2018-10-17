import numpy as np
import pandas as pd
import heapq
import math
import time

from itertools import product, compress
from rule import make_all_ones, make_zeros, rule_vand, rule_vectompz


class CacheTree:
    """
    A tree data structure.
    leaves: a 2-d tuple to encode the leaves
    num_captured: a list to record number of data captured by the leaves
    """

    def __init__(self, lamb, leaves):
        self.leaves = leaves
        self.risk = sum([l.loss for l in leaves])+lamb*len(leaves)

    def sorted_leaves(self):
        # Used by the cache
        return tuple(sorted(leaf.rules for leaf in self.leaves))


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
                       if splitleaf[i] == 0]) + lamb*l

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
            if num_cap_rm<ndata:
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

    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric


class CacheLeaf:
    """
    A data structure to cache every single leaf (symmetry aware)
    """

    def __init__(self, rules, y, z, points_cap, num_captured, lamb, support, is_feature_dead):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured
        self.is_feature_dead = is_feature_dead

        # the y's of these data captured by leaf antecedent[0]
        # y_leaf = y[tag]
        # print("tag",tag)
        # print("y",y)
        _, num_ones = rule_vand(points_cap, rule_vectompz(y))

        # b0 is defined in (28)

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

        self.loss = float(self.num_captured_incorrect) / len(y)

        # Lower bound on antecedent support
        if support:
            # self.is_dead = self.num_captured / len(y) / 2 <= lamb
            self.is_dead = self.loss <= lamb
        else:
            self.is_dead = 0


def log(lines, COUNT_POP, COUNT, queue, metric, R_c, tree_old, tree_new,sorted_new_tree_rules):
    "log"

    the_count_pop = str(COUNT_POP)
    the_count = str(COUNT)
    the_queue_size = str(0)#str(len(queue))
    the_metric = str(metric)
    the_Rc = str(R_c)
    
    the_old_tree = str(sorted([leaf.rules for leaf in tree_old.cache_tree.leaves]))
    the_old_tree_splitleaf = str(tree_old.splitleaf)
    the_old_tree_objective = str(tree_old.cache_tree.risk)
    the_old_tree_lbound = str(tree_old.lb)
    the_new_tree = str(list(sorted_new_tree_rules))
    the_new_tree_splitleaf = str(0)#str(tree_new.splitleaf)
    
    the_new_tree_objective = str(0)#str(tree_new.cache_tree.risk)
    the_new_tree_lbound = str(0)#str(tree_new.lb)
    the_new_tree_length = str(0)#str(len(tree_new.cache_tree.leaves))
    the_new_tree_depth = str(0)#str(max([len(leaf.rules) for leaf in tree_new.leaves]))

    the_queue = str(0)#str([[ leaf.rules for leaf in thetree.leaves]  for _,thetree in queue])
    
    line = ";".join([the_count_pop, the_count, the_queue_size, the_metric, the_Rc,
                     the_old_tree, the_old_tree_splitleaf, the_old_tree_objective, the_old_tree_lbound,
                     the_new_tree, the_new_tree_splitleaf,
                     the_new_tree_objective, the_new_tree_lbound, the_new_tree_length, the_new_tree_depth,
                     the_queue
                    ])
    lines.append(line)


def generate_new_splitleaf(tree_new_leaves, sorted_new_tree_rules, leaf_cache, nleaves, lamb,
                           R_c, accu_support, equiv_points, lookahead):
    """
    generate the new splitleaf for the new tree
    """
    tree_new_rules = [leaf.rules for leaf in tree_new_leaves]
    
    found = False
    for r1 in sorted_new_tree_rules:
        for j in range(len(r1)):
            r2 = tuple(sorted(r1[:j]+(-r1[j],)+r1[j+1:]))
            r0 = r1[:j]+r1[j+1:]
            #print("r1:",r1)
            #print("r2:",r2)
            #print("sorted_tree_new_rules",sorted_tree_new_rules)
            if r2 in sorted_new_tree_rules and r0 in leaf_cache:
                l1 = r1
                l2 = r2
                l0 = r0
                found = True
                break
                #print("l1",l1)
        if found == True:
            break
    
    idx1 = tree_new_rules.index(l1)
    idx2 = tree_new_rules.index(l2)

    loss1 = tree_new_leaves[idx1].loss
    loss2 = tree_new_leaves[idx2].loss
    loss0 = leaf_cache[l0].loss

    lb = sum([leaf.loss for leaf in tree_new_leaves]) - loss1 - loss2 + lamb*(len(tree_new_leaves)-1)
    
    #print("l1",l1)
    #print("l2",l2)
    #print("l0",l0)
    #print("leaf_cache",leaf_cache)
    b0 = leaf_cache[l0].B0


    #(Lower bound on accurate antecedent support)
    #a_l = (sum(cap_l) - sum(incorr_l)) / ndata - sum(cap_l) / ndata / 2
    a_l = loss0 - loss1 - loss2

    if accu_support==False:
        a_l = float('Inf')

    # binary vector indicating split or not
    splitleaf1 = [1] * nleaves  # all leaves labeled as to be split
    splitleaf2 = [0] * (nleaves)# l1,l2 labeled as to be split
    splitleaf2[idx1]=1
    splitleaf2[idx2]=1
    splitleaf3 = [1] * (nleaves)# dp labeled as to be split
    splitleaf3[idx1]=0
    splitleaf3[idx2]=0

    lambbb = lamb
    if lookahead==False:
        lambbb = 0
    
    b00 = b0
    if equiv_points==False:
        b00 = 0

    sl = []

    #print("a_l:",a_l)
    #print("lb + b00 + lambbb",lb + b00 + lambbb )
    #print("R_c",R_c)
    
    if lb + b00 + lambbb >= R_c:
        # print("lb+b0+lamb",lb+b0+lamb)
        # print("R_c",R_c)
        # if equivalent points bound combined with the lookahead bound doesn't hold
        # or if the hierarchical objective lower bound doesn't hold
        # we need to split at least one leaf in dp

        if a_l <= lamb:
            # if the bound doesn't hold, we need to split the leaf l1/l2
            # further

            sl.append(splitleaf2)
            sl.append(splitleaf3)

        else:

            sl.append(splitleaf3)
    else:

        if a_l <= lamb:
            # if the bound doesn't hold, we need to split the leaf l1/l2
            # further

            sl.append(splitleaf2)

        else:
            sl.append(splitleaf1)

    return sl


def gini_reduction(x,y,ndata,nrule):
    """
    calculate the gini reduction by each feature
    return the rank of by descending
    """
    
    p0 = sum(y==1)/ndata
    gini0 = 2*p0*(1-p0)
    
    gr = []
    for i in range(nrule):
        xi = x[:,i]
        y1 = y[xi == 0]
        y2 = y[xi == 1]
        ndata1 = len(y1)
        ndata2 = len(y2)
        p1 = sum(y1==1)/ndata1
        p2 = sum(y2==1)/ndata2
        gini1 = 2*p1*(1-p1)
        gini2 = 2*p2*(1-p2)
        gini_red = gini0 - ndata1/ndata*gini1 - ndata2/ndata*gini2
        gr.append(gini_red)
        
    gr = pd.Series(gr)
    rk = list(map(lambda x: int(x)-1, list(gr.rank(method = 'first'))[::-1])) 
    
    print("the rank of x's columns: ", rk)
    return rk


def bbound_nosimilar_multicopies(x, y, lamb, prior_metric=None, MAXDEPTH=4, niter=float('Inf'), logon=False,
           support=True, accu_support=True, equiv_points=True, lookahead=True):
    """
    An implementation of Algorithm
    ## one copy of tree
    ## mark which leaves to be split
    """

    # Initialize best rule list and objective
    # d_c = None
    # R_c = 1

    nrule = x.shape[1]
    ndata = len(y)
    print("nrule:", nrule)
    print("ndata:", ndata)
    
    # order the columns by descending gini reduction
    idx = gini_reduction(x,y,ndata,nrule)
    x = x[:,idx]

    """
    calculate z, which is for the equivalent points bound
    z is the vector defined in algorithm 5 of the CORELS paper
    z is a binary vector indicating the data with a minority lable in its equivalent set
    """
    z = pd.DataFrame([-1] * ndata).as_matrix()
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
            pred = int(y_l.sum() / len(y_l) >= 0.5)
            # tag2 indicates the samples in a equiv set which have the minority label
            tag2 = (y_l != pred)
            z[tag1, 0] = tag2

    tic = time.time()

    lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf((), y, z, make_all_ones(ndata + 1), ndata, lamb, support, [0] * nrule)
    tree0 = Tree(cache_tree=CacheTree(leaves=[root_leaf], lamb=lamb), lamb=lamb,
                 ndata=ndata, splitleaf=[1], prior_metric=prior_metric)
    d_c = tree0
    R_c = tree0.cache_tree.risk
    #print("Tree0 R_c", R_c)

    heapq.heappush(queue, (tree0.metric, tree0))
    # heapq.heappush(queue, (2*tree0.metric - R_c, tree0))
    # queue.append(tree0)

    # log(lines, lamb, tic, len(queue), tuple(), tree0, R, d_c, R_c)

    leaf_cache[()] = root_leaf

    COUNT = 0  # count the total number of trees in the queue

    COUNT_POP = 0
    while queue and COUNT < niter:
        # tree = queue.pop(0)
        metric, tree = heapq.heappop(queue)

        '''
        if prior_metric == "bound":
            if tree.lb + lamb*len(tree.splitleaf) >= R_c:
                break
        '''

        COUNT_POP = COUNT_POP + 1

        # print([leaf.rules for leaf in tree.leaves])
        # print("curio", curio)
        leaves = tree.cache_tree.leaves

        # print("=======COUNT=======",COUNT)
        # print("d",d)
        # print("R",tree.lbound[0]+(tree.num_captured_incorrect[0])/len(y))

        leaf_split = tree.splitleaf
        leaf_no_split = [not split for split in leaf_split]
        removed_leaves = list(compress(leaves, leaf_split))
        unchanged_leaves = list(compress(leaves, leaf_no_split))

        # lb = sum(l.loss for l in unchanged_leaves)
        # b0 = sum(l.b0 for l in removed_leaves)

        # Generate all assignments of rules to the leaves that are due to be split
        rules_for_leaf = [set(range(1, nrule + 1)) -
                          set(map(abs, l.rules)) for l in removed_leaves]

        for leaf_rules in product(*rules_for_leaf):
            new_leaves = []
            flag_increm = False  # a flag for jump out of the loops (incremental support bound)
            for rule, removed_leaf in zip(leaf_rules, removed_leaves):

                rule_index = rule - 1
                tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf
                parent_is_feature_dead = removed_leaf.is_feature_dead.copy()


                for new_rule in (-rule, rule):
                    new_rule_label = int(new_rule > 0)
                    new_rules = tuple(
                        sorted(removed_leaf.rules + (new_rule,)))
                    if new_rules not in leaf_cache:
                        tag_rule = rule_vectompz(np.array(x[:, rule_index] == new_rule_label) * 1)
                        new_points_cap, new_num_captured = rule_vand(tag, tag_rule)
                        #print("tag:", tag)
                        #print("tag_rule:", tag_rule)
                        #print("new_points_cap:", new_points_cap)
                        #rint("new_num_captured:", new_num_captured)

                        new_leaf = CacheLeaf(new_rules, y, z, new_points_cap, new_num_captured,
                                             lamb, support, parent_is_feature_dead)
                        leaf_cache[new_rules] = new_leaf
                        new_leaves.append(new_leaf)
                    else:
                        new_leaf = leaf_cache[new_rules]
                        new_leaves.append(new_leaf)

                    #print("new_leaf:", new_leaf.rules)
                    #print("leaf loss:", new_leaf.loss)
                    #print("new_leaf.num_captured:",new_leaf.num_captured)
                    #print("new_leaf.num_captured_incorrect",new_leaf.num_captured_incorrect)

                    # incremental support bound
                    if (new_leaf.num_captured - new_leaf.num_captured_incorrect) / ndata <= lamb:
                        removed_leaf.is_feature_dead[rule_index] = 1
                        flag_increm = True
                        break

                if flag_increm:
                    break

            if flag_increm:
                continue

            new_tree_leaves = unchanged_leaves + new_leaves

            sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in new_tree_leaves))



            if logon:
                log(lines, COUNT_POP, COUNT, queue, metric, R_c, tree, tree, sorted_new_tree_rules)



            if sorted_new_tree_rules in tree_cache:
                #print("====== New Tree Duplicated!!! ======")
                #print("sorted_new_tree_rules:", sorted_new_tree_rules)
                continue
            else:
                tree_cache[sorted_new_tree_rules] = True

            child = CacheTree(leaves=new_tree_leaves, lamb=lamb)

            R = child.risk
            #print("child:", child.sorted_leaves())
            #print("R:",R)
            if R < R_c:
                d_c = child
                R_c = R
                C_c = COUNT+1
                time_c = time.time() - tic

            num_unchanged_leaves = len(unchanged_leaves)
            new_tree_length = len(new_tree_leaves)

            # generate the new splitleaf for the new tree
            sl = generate_new_splitleaf(new_tree_leaves, sorted_new_tree_rules, leaf_cache, new_tree_length,
                                        lamb, R_c, accu_support, equiv_points, lookahead)
            #print("sl:", sl)

            # A leaf cannot be split if
            # 1. the MAXDEPTH has been reached
            # 2. the leaf is dead (because of antecedent support)
            # 3. all the features that have not been used are dead
            cannot_split = [len(l.rules) >= MAXDEPTH or l.is_dead or
                            all([l.is_feature_dead[r - 1] for r in range(1, nrule + 1)
                                 if r not in map(abs, l.rules)])
                            for l in new_tree_leaves]
            if len(sl) == 1:
                # For each copy, we don't split leaves which are not split in its parent tree.
                # In this way, we can avoid duplications.
                can_split_leaf = [(0,)]*num_unchanged_leaves +\
                                 [(0,) if cannot_split[i] or sl[0][i] == 0
                                  else (0, 1) for i in range(num_unchanged_leaves, new_tree_length)]
                # Discard the first element of leaf_splits, since we must split at least one leaf
                new_leaf_splits = sorted(product(*can_split_leaf))[1:]
                #print("num_unchanged_leaves:",num_unchanged_leaves)
                #print("cannot_split:", cannot_split)
                #print("can_split_leaf:",can_split_leaf)
                #print("new_leaf_splits:",new_leaf_splits)
            else:
                # For each copy, we don't split leaves which are not split in its parent tree.
                # In this way, we can avoid duplications.
                can_split_leaf = [(0,)] * num_unchanged_leaves + \
                                 [(0,) if cannot_split[i]
                                  else (0, 1) for i in range(num_unchanged_leaves, new_tree_length)]
                # Discard the first element of leaf_splits, since we must split at least one leaf
                new_leaf_splits0 = sorted(product(*can_split_leaf))[1:]
                # Filter out those which split at least one leaf in dp and split at least one leaf in d0
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if sum(np.array(ls)*sl[0]) > 0 and sum(np.array(ls)*sl[1]) > 0]

            for new_leaf_split in new_leaf_splits:
                # construct the new tree
                tree_new = Tree(cache_tree=child, ndata=ndata, lamb=lamb,
                                splitleaf=new_leaf_split, prior_metric=prior_metric)

                # heapq.heappush(queue, (2*tree_new.metric - R_c, tree_new))
                heapq.heappush(queue, (tree_new.metric, tree_new))
                COUNT = COUNT+1

                #if logon:
                #    log(lines, COUNT_POP, COUNT, queue, metric, R_c, tree, tree_new, sorted_new_tree_rules)

                if COUNT % 100000 == 0:
                    print("COUNT:", COUNT)

    header = ['#pop', '#push', 'queue_size', 'metric', 'R_c',
              'the_old_tree', 'the_old_tree_splitleaf', 'the_old_tree_objective', 'the_old_tree_lbound',
              'the_new_tree', 'the_new_tree_splitleaf',
              'the_new_tree_objective', 'the_new_tree_lbound', 'the_new_tree_length', 'the_new_tree_depth', 'queue']

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
    # print("lbound: ", d_c.cache_tree.lbound)
    # print("d_c.num_captured: ", [leaf.num_captured for leaf in d_c.cache_tree.leaves])
    print("prediction: ", [leaf.prediction for leaf in d_c.leaves])
    print("Objective: ", R_c)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return d_c