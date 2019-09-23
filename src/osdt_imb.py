import numpy as np
import pandas as pd
import heapq
import math
import time
import copy
import sklearn.tree
import sklearn.metrics
from itertools import product, compress
from gmpy2 import mpz
from rule import make_all_ones, rule_vand, rule_vectompz, count_ones
from osdt_sup import log, gini_reduction, get_code, cart, get_z

#from matplotlib import pyplot as plt
#import pickle

class Objective: 
    def __init__(self, name): # name ="acc", "bacc", "wacc", "auc", "f1"
        self.name = name
    
    def loss(self, P, N, FP, FN, w=None):
        '''
        Input:
            P, N: number of positive (negative) observations in the whole data
            FP, FN: false positve (negative) of the tree
        Output:
            f: f(FP, FN) or l(d,x,y)
        '''       
        if self.name == 'acc':
            f = (FP+FN)/(P+N)
        elif self.name == 'bacc':
            f = 0.5*(FN/P + FP/N)
        elif self.name == 'wacc':
            f = (FP+w*FN)/(P+N)
        elif self.name == 'auc':
            f = (P*FP + N*FN)/(2*P*N)
        elif self.name == 'f1':
            f = (FP+FN)/(2*P+FP-FN)
        return f
    
    def leaf_predict(self, P, N, p, n, w=None):
        predict = 1
        if self.name == 'acc':
            if p<n:
                predict = 0
        elif self.name == 'bacc':
            if p/P <= n/N:
                predict = 0
        elif self.name == 'wacc':
            if p/(p+n) <= 1/(1+w):
                predict = 0
        elif self.name == 'auc':
            if N*p <= P*n:
                predict = 0
        elif self.name == 'f1':
            if w*p <= n:
                predict = 0
        
        return predict

class CacheTree:
    def __init__(self, name, P, N, lamb, leaves, w=None):
        self.name = name
        self.P = P
        self.N = N
        self.leaves = leaves
        self.H = len(leaves)
        self.FP = sum([l.fp for l in leaves])
        self.FN = sum([l.fn for l in leaves])
        self.w = w
        
        if name != 'auc_convex':
            bound = Objective(name)
            self.risk = bound.loss(self.P, self.N, self.FP, self.FN, self.w) + lamb*self.H
        elif name == 'auc_convex':
            _, _, _, loss = convex_hull(self.leaves, self.P, self.N)
            self.risk = loss+lamb*self.H
    
    def sorted_leaves(self):
        return tuple(sorted(leaf.rules for leaf in self.leaves))
            
        

class Tree:
    def __init__(self, cache_tree, n, lamb, splitleaf=None, prior_metric=None):
        
        self.cache_tree = cache_tree
        self.splitleaf = splitleaf
        self.H = cache_tree.H
        leaves = cache_tree.leaves
        
        if cache_tree.name != 'auc_convex':
            self.FPu = sum([leaves[i].fp for i in range(self.H) if splitleaf[i]==0])
            self.FNu = sum([leaves[i].fn for i in range(self.H) if splitleaf[i]==0])
            bound = Objective(cache_tree.name)
            self.lb = bound.loss(cache_tree.P, cache_tree.N, self.FPu, self.FNu, cache_tree.w) + lamb*self.H
        elif cache_tree.name == 'auc_convex':
            split_set = []
            split_set.append([leaves[i] for i in range(self.H) if splitleaf[i]==1])
            split_p = sum([leaves[i].p for i in range(self.H) if splitleaf[i]==1])
            split_n = sum([leaves[i].n for i in range(self.H) if splitleaf[i]==1])
            ordered_leaves, _, _, _ = convex_hull(leaves, cache_tree.P, cache_tree.N)
            tp = np.array([0, split_p])
            fp = np.array([0, 0])
            ordered_fixed = [l for l in ordered_leaves if l not in split_set]
            for i in range(0, len(ordered_fixed)):
                tp = np.append(tp, tp[i+1]+ordered_fixed[i].p)
                fp = np.append(fp, fp[i+1]+ordered_fixed[i].n)
            tp = np.append(tp, tp[len(ordered_fixed)+1]+0)
            fp = np.append(fp, fp[len(ordered_fixed)+1]+split_n)
            self.lb = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(cache_tree.P*cache_tree.N) for i in range(1,len(tp))]) + lamb*self.H
            
            
        if leaves[0].num_captured == n:
            self.metric = 0                #null tree
        elif prior_metric == "objective":
            self.metric = self.risk
        elif prior_metric == "bound":
            self.metric = self.lb
        elif prior_metric == "curiosity":
            removed_leaves = list(compress(leaves, splitleaf)) #dsplit
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves) # num captured by dsplit
            if num_cap_rm < n:
                self.metric = self.lb / ((n - num_cap_rm) / n) # supp(dun, xn)
            else:
                self.metric = self.lb / (0.01 / n) # null tree
        elif prior_metric == "entropy":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # entropy weighted by number of points captured
            self.entropy = [
                (-leaves[i].p * math.log2(leaves[i].p) - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)) * leaves[
                    i].num_captured if leaves[i].p != 0 and leaves[i].p != 1 else 0 for i in range(self.H)]
            if num_cap_rm < n:
                self.metric = sum(self.entropy[i] for i in range(self.H) if splitleaf[i] == 0) / (
                        n - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.entropy[i] for i in range(self.H) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "gini":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # gini index weighted by number of points captured
            self.giniindex = [(2 * leaves[i].p * (1 - leaves[i].p))
                              * leaves[i].num_captured for i in range(self.H)]
            if num_cap_rm < n:
                self.metric = sum(self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0) / (
                        n - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "FIFO":
            self.metric = 0
        
    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric
    

class CacheLeaf:
    def __init__(self, name, n, P, N, rules, y_mpz, z_mpz, points_cap, 
                 num_captured, lamb, support, is_feature_dead, w=None):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured
        self.is_feature_dead = is_feature_dead

        _, num_ones = rule_vand(points_cap, y_mpz) #return vand and cnt
        _, num_errors = rule_vand(points_cap, z_mpz)
        
        self.delta = num_errors
        self.p = num_ones
        self.n = self.num_captured - num_ones
        if self.num_captured > 0 :
            self.r = num_ones/self.num_captured
        else:
            self.r = 0
        bound = Objective(name)
        
        if self.num_captured:
            self.pred = bound.leaf_predict(P, N, self.p, self.n, w)
            if self.pred == 0:
                self.fp = 0
                self.fn = self.p
                self.delta_fp = 0
                self.delta_fn = self.delta
            else:
                self.fp = self.n
                self.fn = 0
                self.delta_fp = self.delta
                self.delta_fn = 0
        else:
            self.pred = 0
            self.fp = 0
            self.fn = self.p
        
        if name == 'acc':
            loss = float(self.fp+self.fn) / n 
            if support:
                # self.is_dead = self.num_captured / len(y) / 2 <= lamb
                self.is_dead = loss <= lamb
            else:
                self.is_dead = 0
            
def convex_hull(leaves, P, N):
    ordered_leaves = sorted([l for l in leaves], key=lambda x:x.r, reverse=True)
    tp = fp = np.array([0])
    if len(leaves) > 1:
        for i in range(0, len(leaves)):
            tp = np.append(tp, tp[i]+ordered_leaves[i].p)
            fp = np.append(fp, fp[i]+ordered_leaves[i].n)
    else:
        tp = np.append(tp, P)
        fp = np.append(fp, N)
        
    loss = 1-0.5*sum([(tp[i]/P+tp[i-1]/P)*(fp[i]/N-fp[i-1]/N) for i in range(1,len(tp))])
    return ordered_leaves, tp, fp, loss
    
def upper_bound_H(R_c, lamb, m):
    '''
    R_c: current best risk 
    m: number of features
    '''
    return min(math.floor(R_c/lamb), 2**m)

def upper_bound_H_child(R_c, lamb, m, lb, H):
    '''
    R_c: current best risk 
    m: number of features
    lb: lower bound of the tree, b(d,x,y)
    H: number of leaves of the parent tree
    '''
    return min(H + math.floor((R_c - lb) / lamb), 2**m)


def generate_new_splitleaf(name, P, N, unchanged_leaves, removed_leaves, new_leaves, lamb,
                           R_c, incre_support, w):
    """
    generate the new splitleaf for the new tree
    """

    n_removed_leaves = len(removed_leaves)  #dsplit
    n_unchanged_leaves = len(unchanged_leaves) #dun
    n_new_leaves = len(new_leaves)

    n_new_tree_leaves = n_unchanged_leaves + n_new_leaves #H'

    splitleaf1 = [0] * n_unchanged_leaves + [1] * n_new_leaves  # all new leaves labeled as to be split
    
    if name == "f1":
        FPu = sum([l.fp for l in unchanged_leaves])
        FNu = sum([l.fn for l in unchanged_leaves])
    
    if name == 'auc_convex':
        ordered_leaves, tp, fp, ld = convex_hull(unchanged_leaves+removed_leaves, P, N)

    sl = []
    for i in range(n_removed_leaves):

        splitleaf = [0] * n_new_tree_leaves

        idx1 = 2*i
        idx2 = 2*i+1
        
        # (Lower bound on incremental classification accuracy)
        bound = Objective(name)
        if (name != 'f1') & (name != 'auc_convex'):
            a = bound.loss(P, N, removed_leaves[i].fp-new_leaves[idx1].fp-new_leaves[idx2].fp, 
                           removed_leaves[i].fn-new_leaves[idx1].fn-new_leaves[idx2].fn, w)
        elif name == 'f1':
            a = bound.loss(P, N, FPu+removed_leaves[i].fp, FNu+removed_leaves[i].fn, w) -\
            bound.loss(P, N, FPu+new_leaves[idx1].fp+new_leaves[idx2].fp, 
                                FNu+new_leaves[idx1].fn+new_leaves[idx2].fn, w)
        elif name == 'auc_convex':
            removed = removed_leaves.copy()
            removed.remove(removed_leaves[i])
            ordered_leaves, tp, fp, ld_new = convex_hull(unchanged_leaves+removed+
                                         [new_leaves[idx1]]+[new_leaves[idx2]], P, N)
            a = ld-ld_new

        if not incre_support:
            a = float('Inf')

        if a <= lamb:
            splitleaf[n_unchanged_leaves + idx1] = 1
            splitleaf[n_unchanged_leaves + idx2] = 1
            sl.append(splitleaf)
        else:
            sl.append(splitleaf1)

    return sl


def bbound(x, y, name, lamb, prior_metric=None, w=None, MAXDEPTH=float('Inf'), 
           MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=True, R_c0 = 1, timelimit=float('Inf'), init_cart = True,
           saveTree = False, readTree = False):

    x0 = copy.deepcopy(x)
    y0 = copy.deepcopy(y)

    tic = time.time()

    m = x.shape[1] # number of features
    n = len(y)
    P = np.count_nonzero(y)
    N = n-P

    x_mpz = [rule_vectompz(x[:, i]) for i in range(m)]
    y_mpz = rule_vectompz(y)

    # order the columns by descending gini reduction
    idx, dic = gini_reduction(x_mpz, y_mpz, n, range(m))
    x = x[:, idx]
    x_mpz = [x_mpz[i] for i in idx]
    
    z_mpz = get_z(x,y,n,m)


    #lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf(name, n, P, N, (), y_mpz, z_mpz, make_all_ones(n + 1), 
                          n, lamb, support, [0] * m, w)
    d_c = CacheTree(name, P, N, lamb=lamb, leaves=[root_leaf], w=w)
    R_c = d_c.risk
    tree0 = Tree(cache_tree=d_c, n=n, lamb=lamb,splitleaf=[1], prior_metric=prior_metric)
    heapq.heappush(queue, (tree0.metric, tree0))
    
    best_is_cart = False  # a flag for whether or not the best is the initial CART
    if init_cart: 
        clf, nleaves_CART, trainout_CART, R_c, d_c, C_c = cart(x0, y0, name, n, lamb, w, MAXDEPTH)
        time_c = time.time() - tic
        best_is_cart = True
    else:
        C_c=0
        clf=None
        time_c = time.time()
        
    if R_c0 < R_c:
        R_c = R_c0

    leaf_cache[()] = root_leaf

    COUNT = 0  # count the total number of trees in the queue
    COUNT_POP = 0
    COUNT_UNIQLEAVES = 0
    COUNT_LEAFLOOKUPS = 0
    
    bound = Objective(name)

    while queue and COUNT < niter and time.time() - tic < timelimit:
        metric, tree = heapq.heappop(queue)

        COUNT_POP = COUNT_POP + 1
        
        leaves = tree.cache_tree.leaves
        leaf_split = tree.splitleaf
        
        removed_leaves = list(compress(leaves, leaf_split))
        new_tree_length = len(leaf_split) + sum(leaf_split)
        
        # prefix-specific upper bound on number of leaves
        if lenbound and new_tree_length >= upper_bound_H_child(R_c, lamb, m, tree.lb, tree.H):
            continue

        n_removed_leaves = sum(leaf_split)
        n_unchanged_leaves = tree.H - n_removed_leaves
        
        '''equivalent points bound + lookahead bound'''
        delta_fp = sum([leaf.delta_fp for leaf in removed_leaves]) if equiv_points else 0
        delta_fn = sum([leaf.delta_fn for leaf in removed_leaves]) if equiv_points else 0
        lambbb = lamb if lookahead else 0
        
        if (name != 'auc_convex') and (bound.loss(P, N, tree.FPu+delta_fp, tree.FNu+delta_fn, w) + n_removed_leaves * lambbb >= R_c):
            continue
        
        if (name == 'auc_convex') and (tree.lb >= R_c):
            continue


        leaf_no_split = [not split for split in leaf_split]
        unchanged_leaves = list(compress(leaves, leaf_no_split))

        # Generate all assignments of rules to the leaves that are due to be split
        rules_for_leaf = [set(range(1, m + 1)) - set(map(abs, l.rules)) -
                          set([i+1 for i in range(m) if l.is_feature_dead[i] == 1]) for l in removed_leaves]


        for leaf_rules in product(*rules_for_leaf):

            if time.time() - tic >= timelimit:
                break

            new_leaves = []
            flag_increm = False  # a flag for jump out of the loops (incremental support bound)
            for rule, removed_leaf in zip(leaf_rules, removed_leaves):

                rule_index = rule - 1
                tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf

                for new_rule in (-rule, rule):
                    new_rule_label = int(new_rule > 0)
                    new_rules = tuple(
                        sorted(removed_leaf.rules + (new_rule,)))
                    if new_rules not in leaf_cache:

                        COUNT_UNIQLEAVES = COUNT_UNIQLEAVES+1

                        tag_rule = x_mpz[rule_index] if new_rule_label == 1 else ~(x_mpz[rule_index]) | mpz(pow(2, n))
                        new_points_cap, new_num_captured = rule_vand(tag, tag_rule)

                        #parent_is_feature_dead =
                        new_leaf = CacheLeaf(name, n, P, N, new_rules, y_mpz, z_mpz, new_points_cap, new_num_captured,
                                             lamb, support, removed_leaf.is_feature_dead.copy(), w)
                        leaf_cache[new_rules] = new_leaf
                        new_leaves.append(new_leaf)
                    else:

                        COUNT_LEAFLOOKUPS = COUNT_LEAFLOOKUPS+1

                        new_leaf = leaf_cache[new_rules]
                        new_leaves.append(new_leaf)

                    '''
                    # Lower bound on classification accuracy
                    # if (new_leaf.num_captured) / n <= lamb:
                    # accu_support == theorem 9 in OSDT, check if feature dead, not derived yet
                    
                    if accu_support == True and (new_leaf.num_captured - new_leaf.num_captured_incorrect) / n <= lamb:

                        removed_leaf.is_feature_dead[rule_index] = 1

                        flag_increm = True
                        break
                    '''    

                if flag_increm:
                    break

            if flag_increm:
                continue

            new_tree_leaves = unchanged_leaves + new_leaves

            sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in new_tree_leaves))

            if sorted_new_tree_rules in tree_cache:
                continue
            else:
                tree_cache[sorted_new_tree_rules] = True

            #child = CacheTree(leaves=new_tree_leaves, lamb=lamb)
            '''
            R = bound.loss(P, N, sum([l.fp for l in new_tree_leaves]), 
                       sum([l.fn for l in new_tree_leaves]), w) + lamb * len(new_tree_leaves)
            '''
            child = CacheTree(name, P, N, lamb, new_tree_leaves, w=w)

            R = child.risk
            if R < R_c:
                d_c = child
                R_c = R
                C_c = COUNT + 1
                time_c = time.time() - tic

                best_is_cart = False

            # generate the new splitleaf for the new tree
            sl = generate_new_splitleaf(name, P, N, unchanged_leaves, removed_leaves, new_leaves,
                                        lamb, R_c, incre_support, w) # incre_support is a_j
            # print("sl:", sl)
            ''' 
            # A leaf cannot be split if
            # 1. the MAXDEPTH has been reached
            # 2. the leaf is dead (because of antecedent support)
            # 3. all the features that have not been used are dead
            return a list of true false
            cannot_split = [len(l.rules) >= MAXDEPTH or l.is_dead or
                            all([l.is_feature_dead[r - 1] for r in range(1, m + 1)
                                 if r not in map(abs, l.rules)])
                            for l in new_tree_leaves]
                            
            '''
            if name == 'acc':
                cannot_split = [len(l.rules) >= MAXDEPTH or l.is_dead or
                                all([l.is_feature_dead[r - 1] for r in range(1, m + 1)
                                     if r not in map(abs, l.rules)])
                                for l in new_tree_leaves]
            elif name == 'f1':
                FP = sum([l.fp for l in new_tree_leaves])
                FN = sum([l.fn for l in new_tree_leaves])
                loss = bound.loss(P, N, FP, FN, w)
                
                cannot_split = [len(l.rules) >= MAXDEPTH or 
                                loss-bound.loss(P,N, FP-l.fp, FN-l.fn) < lamb or
                                all([l.is_feature_dead[r - 1] for r in range(1, m + 1)
                                 if r not in map(abs, l.rules)])
                                for l in new_tree_leaves]
            elif name == 'auc_convex':
                ordered_leaves, tp, fp, loss = convex_hull(new_tree_leaves, P, N)
                
                cannot_split = []
                for l in new_tree_leaves:
                    if l.p==0 or l.n == 0:
                        cannot_split.append(True)
                    else:
                        leaves = ordered_leaves.copy()
                        leaves.remove(l)
                        tp = fp = np.array([0])
                        tp = np.append(tp, tp[0]+l.p)
                        fp = np.append(fp, fp[0]+0)
                        for i in range(0, len(leaves)):
                            tp = np.append(tp, tp[i+1]+leaves[i].p)
                            fp = np.append(fp, fp[i+1]+leaves[i].n)
                        tp = np.append(tp, tp[len(leaves)+1]+0)
                        fp = np.append(fp, fp[len(leaves)+1]+l.n)

                        loss_new = 1- 0.5*sum([(tp[i]/P+tp[i-1]/P)*(fp[i]/N-fp[i-1]/N) for i in range(1,len(tp))])
                        delta_loss = loss - loss_new
                        if len(l.rules)>=MAXDEPTH or delta_loss<lamb or all([l.is_feature_dead[r - 1] for r in range(1, m + 1) if r not in map(abs, l.rules)]):
                            cannot_split.append(True)
                        else:
                            cannot_split.append(False)
            else:
                cannot_split = [len(l.rules) >= MAXDEPTH or bound.loss(P, N, l.fp, l.fn, w) < lamb or 
                                all([l.is_feature_dead[r - 1] for r in range(1, m + 1)
                                 if r not in map(abs, l.rules)])
                                for l in new_tree_leaves]
                


            # For each copy, we don't split leaves which are not split in its parent tree.
            # In this way, we can avoid duplications.
            can_split_leaf = [(0,)] * n_unchanged_leaves + \
                             [(0,) if cannot_split[i]
                              else (0, 1) for i in range(n_unchanged_leaves, new_tree_length)]
            # Discard the first element of leaf_splits, since we must split at least one leaf
            new_leaf_splits0 = np.array(list(product(*can_split_leaf))[1:])#sorted(product(*can_split_leaf))[1:]
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
                tree_new = Tree(cache_tree=child, n=n, lamb=lamb,
                                splitleaf=new_leaf_split, prior_metric=prior_metric)
                
                '''
                tree_new = CacheTree(name, n, P, N, lamb, new_tree_leaves, new_leaf_split, prior_metric, w)
                '''
                # MAX Number of leaves
                if len(new_leaf_split)+sum(new_leaf_split) > MAX_NLEAVES:
                    continue

                COUNT = COUNT + 1
                # heapq.heappush(queue, (2*tree_new.metric - R_c, tree_new))
                heapq.heappush(queue, (tree_new.metric, tree_new))
                
                if COUNT % 1000000 == 0:
                    print("COUNT:", COUNT)

    totaltime = time.time() - tic

    if not best_is_cart:

        accu = 1-(R_c-lamb*len(d_c.leaves))

        leaves_c = [leaf.rules for leaf in d_c.leaves]
        pred_c = [leaf.pred for leaf in d_c.leaves]

        num_captured = [leaf.num_captured for leaf in d_c.leaves]

        #num_captured_incorrect = [leaf.num_captured_incorrect for leaf in d_c.leaves]

        nleaves = len(leaves_c)
    else:
        accu = trainout_CART
        leaves_c = 'NA'
        pred_c = 'NA'
        get_code(d_c, ['x'+str(i) for i in range(1, m+1)], [0, 1])
        num_captured = 'NA'
        #num_captured_incorrect = 'NA'
        nleaves = nleaves_CART

    '''
    print(">>> log:", logon)
    print(">>> support bound:", support)
    print(">>> accu_support:", accu_support)
    print(">>> accurate support bound:", incre_support)
    print(">>> equiv points bound:", equiv_points)
    print(">>> lookahead bound:", lookahead)
    print("prior_metric=", prior_metric)
    '''
    print("lambda: ", lamb)
    print("COUNT_UNIQLEAVES:", COUNT_UNIQLEAVES)
    print("COUNT_LEAFLOOKUPS:", COUNT_LEAFLOOKUPS)

    print("total time: ", totaltime)
    
    print("leaves: ", leaves_c)
    print("num_captured: ", num_captured)
    #print("num_captured_incorrect: ", num_captured_incorrect)
    # print("lbound: ", d_c.cache_tree.lbound)
    # print("d_c.num_captured: ", [leaf.num_captured for leaf in d_c.cache_tree.leaves])
    print("prediction: ", pred_c)
    print("Objective: ", R_c)
    print("Accuracy: ", accu)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return leaves_c, pred_c, dic, nleaves, m, n, totaltime, time_c, COUNT, C_c, accu, best_is_cart, clf


def predict(name, leaves_c, prediction_c, dic, x, y, best_is_cart, clf, w=None):
    """
    :param leaves_c:
    :param dic:
    :return:
    """
    P = np.count_nonzero(y)
    N = len(y) - P
    if best_is_cart:
        yhat = clf.predict(x)
        
        n_fp = sum((yhat == 1) & (yhat != y))
        n_fn = sum((yhat == 0) & (yhat != y))
        if name == 'acc':
            out = sklearn.metrics.accuracy_score(y, yhat)
        elif name == "bacc":
            out = sklearn.metrics.balanced_accuracy_score(y, yhat)
        elif name == 'wacc':
            n_fp = sum((yhat == 1) & (yhat != y))
            n_fn = sum((yhat == 0) & (yhat != y))
            out = (n_fp + w*n_fn)/(len(y))
        elif (name == 'auc') or (name == 'auc_convex'):
            out = sklearn.metrics.roc_auc_score(y, yhat)
        elif name == 'f1':
            out = sklearn.metrics.f1_score(y, yhat)
        
        print("Best is cart! Testing", name, ":", round(out,4))
        print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
        print(">>>>>>>>>>>>>>>>>>>>>>>")

        return yhat, out

    n = x.shape[0]

    caps = []

    for leaf in leaves_c:
        cap = np.array([1] * n)
        for feature in leaf:
            idx = dic[abs(feature)]
            feature_label = int(feature > 0)
            cap = (x[:, idx] == feature_label) * cap
        caps.append(cap)

    yhat = np.array([1] * n)

    for j in range(len(caps)):
        idx_cap = [i for i in range(n) if caps[j][i] == 1]
        yhat[idx_cap] = prediction_c[j]

    
    n_fp = sum((yhat == 1) & (yhat != y))
    n_fn = sum((yhat == 0) & (yhat != y))
    if name == 'acc':
        out = sklearn.metrics.accuracy_score(y, yhat)
    elif name == "bacc":
        out = sklearn.metrics.balanced_accuracy_score(y, yhat)
    elif name == 'wacc':
        n_fp = sum((yhat == 1) & (yhat != y))
        n_fn = sum((yhat == 0) & (yhat != y))
        out = (n_fp + w*n_fn)/(len(y))
    elif name == 'auc':
        out = sklearn.metrics.roc_auc_score(y, yhat)
    elif name == 'f1':
        out = sklearn.metrics.f1_score(y, yhat)
    elif name == 'auc_convex':
        out = 1234

    print("Testing", name, ":", round(out,4))
    print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return yhat, out
