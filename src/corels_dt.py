import numpy as np
import heapq
import math
import time

from itertools import product, compress
from rule import make_all_ones, rule_vand, rule_vectompz


class Tree:

    def __init__(self, x, y, leaves, lam, prior_metric=None):
        self.leaves = leaves
        self.loss = sum(l.loss for l in leaves) + lam * len(leaves)

        # TODO: add other indexes. Use the gini index for now
        ndata = x.shape[0]
        giniindex = [(2 * l.p * (1 - l.p)) * l.num_captured for l in leaves]
        self.metric = min([sum(giniindex[:i] + giniindex[i + 1:]) / (ndata - leaves[i].num_captured)
                           if ndata - leaves[i].num_captured != 0 else 0 for i in range(len(leaves))])

    def __lt__(self, other):
        # Used by the priority queue
        return self.metric < other.metric

    def sorted_leaves(self):
        return tuple(sorted(leaf.rules for leaf in self.leaves))


class Leaf:

    def __init__(self, rules, points_cap, num_captured, y, z, lam):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured

        _, num_ones = rule_vand(points_cap, rule_vectompz(y))
        if num_captured > 0:
            self.prediction = int(num_ones / num_captured >= .5)
            self.num_captured_incorrect = num_captured - \
                num_ones if self.prediction == 1 else num_ones
            self.p = self.num_captured_incorrect / num_captured
        else:
            self.prediction = 0
            self.num_captured_incorrect = 0
            self.p = 0

        self.loss = float(self.num_captured_incorrect) / len(y)

        # b0 is defined in (28)
        _, num_errors = rule_vand(points_cap, rule_vectompz(z))
        self.b0 = num_errors / len(y)

        # Lower bound on antecedent support
        self.is_dead = num_captured / len(y) / 2 < lam

    def __hash__(self):
        # Used by the cache
        return hash(self.rules)

    def __eq__(self, other):
        # Used by the cache
        return hash(self) == hash(other)


def bbound(x, y, z, lam, prior_metric=None, MAX_DEPTH=4, niter=float('Inf')):
    ndata, nrules = x.shape

    tree_cache = {}
    leaf_cache = {}
    queue = []

    root_leaf = Leaf((), make_all_ones(ndata + 1), ndata, y, z, lam)
    leaf_cache[()] = root_leaf
    root_tree = Tree(x, y, [root_leaf], lam, prior_metric)

    best_tree = root_tree
    best_loss = root_tree.loss
    best_idx = 0

    count = 0
    heapq.heappush(queue, (root_tree.metric, root_tree))

    while len(queue) > 0 and count < niter:
        _, tree = heapq.heappop(queue)
        if tree.sorted_leaves() in tree_cache:
            continue
        else:
            tree_cache[tree.sorted_leaves()] = True

        # Discard the first element of leaf_splits, since we must split at
        # least one leaf
        can_split_leaf = [(0,) if len(l.rules) >= MAX_DEPTH or l.is_dead else (
            0, 1) for l in tree.leaves]
        leaf_splits = sorted(product(*can_split_leaf), key=sum)[1:]

        while len(leaf_splits) > 0:
            leaf_split, leaf_splits = leaf_splits[0], leaf_splits[1:]
            leaf_no_split = [not split for split in leaf_split]
            removed_leaves = list(compress(tree.leaves, leaf_split))
            unchanged_leaves = list(compress(tree.leaves, leaf_no_split))

            lb = sum(l.loss for l in unchanged_leaves)
            b0 = sum(l.b0 for l in removed_leaves)

            if lb + b0 + lam * len(removed_leaves) < best_loss:
                # Eliminate all leaf splits that are covered by the current
                # split
                leaf_splits = [o for o in leaf_splits if not np.array_equal(
                    np.logical_and(leaf_split, o), leaf_split)]

                # Generate all assignments of rules to the leaves that are due to be split,
                # omitting any assigments that duplicate a rule in a path to
                # the leaf
                rules_for_leaf = [set(range(1, nrules + 1)) -
                                  set(map(abs, l.rules)) for l in removed_leaves]
                for leaf_rules in product(*rules_for_leaf):
                    new_leaves = []
                    for rule, removed_leaf in zip(leaf_rules, removed_leaves):
                        for new_rule in (-rule, rule):
                            new_rules = tuple(
                                sorted(removed_leaf.rules + (new_rule,)))
                            if new_rules not in leaf_cache:
                                new_rule_index = abs(new_rule) - 1
                                new_rule_label = int(new_rule > 0)
                                new_rule_points_cap = rule_vectompz(
                                    np.array(x.iloc[:, new_rule_index] == new_rule_label, dtype=int))
                                points_cap, num_captured = rule_vand(
                                    removed_leaf.points_cap, new_rule_points_cap)
                                new_leaf = Leaf(new_rules, points_cap,
                                                num_captured, y, z, lam)
                                new_leaves.append(new_leaf)
                            else:
                                new_leaves.append(leaf_cache[new_rules])

                    child = Tree(x, y, unchanged_leaves +
                                 new_leaves, lam, prior_metric)
                    if child.sorted_leaves() in tree_cache:
                        continue

                    if child.loss < best_loss:
                        best_loss = child.loss
                        best_tree = child
                        best_idx = count

                        print('*' * 20)
                        print('Loss: ', best_loss)
                        print('Count: ', count)
                        print('Tree:')
                        for leaf in best_tree.leaves:
                            leaf_rules = [('' if r > 0 else '¬') +
                                          x.columns[abs(r) - 1] for r in leaf.rules]
                            print('({}): {}'.format(
                                ', '.join(leaf_rules), leaf.prediction))

                    heapq.heappush(queue, (child.metric, child))
                    count += 1
                    if count % 1000 == 0:
                        print('*' * 20)
                        print("Count: ", count)

    return best_tree
