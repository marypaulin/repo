import numpy as np
import heapq
import math
import time

from itertools import product, compress
from rule import make_all_ones, rule_vand, rule_vectompz


class Tree:

    def __init__(self, x, y, leaves, lam, prior_metric=None):
        self.leaves = leaves
        
        ndata = x.shape[0]
        self.loss = sum(l.loss for l in leaves) + lam * len(leaves)
        
        # TODO: add other indexes. Use the gini index for now
        giniindex = [(2 * l.p * (1 - l.p)) * l.num_captured for l in leaves]
        self.metric = min([sum(giniindex[:i] + giniindex[i + 1:]) / (ndata - leaves[i].num_captured)
                           if ndata - leaves[i].num_captured != 0 else 0 for i in range(len(leaves))])

    def __lt__(self, other):
        # Used by the priority queue
        return self.metric < other.metric

    def __hash__(self):
        # Used by the cache
        return hash(tuple(sorted(leaf.rules for leaf in self.leaves)))

    def __eq__(self, other):
        # Used by the cache
        return hash(self) == hash(other)

    # TODO: recreate tree structure from sorted leaves
    # def _to_nested_dict(self):
    #     tree = {}

    #     for i, leaf in enumerate(self.leaves):
    #         current_node = tree

    #         for rule in leaf:
    #             current_node['rule'] = abs(rule)
    #             direction = 'left' if rule < 0 else 'right'
    #             if direction not in current_node:
    #                 current_node[direction] = {}
    #             current_node = current_node[direction]

    #         current_node['label'] = self.prediction[i]

    #     return tree

    # def _format_dict(self, tree, depth=0):
    #     fmt = '-' * depth

    #     if 'rule' in tree:
    #         fmt += 'r{}'.format(tree['rule'])
    #     else:
    #         assert 'label' in tree
    #         fmt += str(tree['label'])

    #     fmt += '\n'

    #     if 'left' in tree:
    #         fmt += self._format_dict(tree['left'], depth + 1)

    #     if 'right' in tree:
    #         fmt += self._format_dict(tree['right'], depth + 1)

    #     return fmt

    # def __str__(self):
    #     return self._format_dict(self._to_nested_dict())


class Leaf:

    def __init__(self, rules, points_cap, num_captured, x, y, z, lam):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured

        # b0 is defined in (28)
        tag_z = rule_vectompz(z.reshape(1, -1)[0])
        _, num_errors = rule_vand(points_cap, tag_z)
        self.b0 = num_errors / len(y)

        _, num_ones = rule_vand(points_cap, rule_vectompz(y))
        if num_captured > 0:
            self.prediction = int(num_ones / num_captured >= .5)
            self.num_captured_incorrect = num_captured - num_ones if self.prediction == 1 else num_ones
            self.p = self.num_captured_incorrect / num_captured
        else:
            self.prediction = 0
            self.num_captured_incorrect = 0
            self.p = 0
        
        self.loss = float(self.num_captured_incorrect) / x.shape[0]

        # Lower bound on antecedent support
        self.is_dead = num_captured / x.shape[0] / 2 < lam

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

    root_leaf = Leaf((), make_all_ones(ndata + 1), ndata, x, y, z, lam)
    leaf_cache[()] = root_leaf
    root_tree = Tree(x, y, [root_leaf], lam, prior_metric)

    best_tree = root_tree
    best_loss = root_tree.loss
    best_idx = 0

    count = 0
    heapq.heappush(queue, (root_tree.metric, root_tree))

    while len(queue) > 0 and count < niter:
        _, tree = heapq.heappop(queue)

        if tree in tree_cache:
            continue
        else:
            tree_cache[tree] = True

        # Discard the first element, since we must split at least one leaf
        leaf_splits = np.array(sorted(list(product([0, 1], repeat=len(tree.leaves))), key=sum)).astype(bool)[1:]

        while len(leaf_splits) > 0:
            leaf_split, leaf_splits = leaf_splits[0], leaf_splits[1:]
            leaf_no_split = np.invert(leaf_split)

            if any(len(l.rules) >= MAX_DEPTH for l in compress(tree.leaves, leaf_split)):
                continue

            if any(l.is_dead for l in compress(tree.leaves, leaf_split)):
                continue

            lb = sum(l.loss for l in compress(tree.leaves, leaf_no_split))
            b0 = sum(l.b0 for l in compress(tree.leaves, leaf_split))

            if lb + b0 + lam < best_loss or lb < best_loss:
                # Eliminate all leaf splits covered by the current split
                leaf_splits = np.array([])
                for other_split in leaf_splits:
                    if np.not_equal(np.logical_and(leaf_split, other_split), leaf_split):
                        leaf_splits = np.stack(leaf_splits, other_split)

                # Generate all assignments of rules to the leaves that are due
                # to be split
                rs = product(range(1, nrules + 1), repeat=np.sum(leaf_split))

                for leaf_assignment in rs:
                    removed_leaves = list(compress(tree.leaves, leaf_split))
                    unchanged_leaves = list(compress(tree.leaves, leaf_no_split))

                    # Eliminate any assigments that duplicate a rule in a
                    # path to the leaf
                    if any(a in l.rules for a, l in zip(leaf_assignment, removed_leaves)):
                        continue

                    new_leaves = []

                    for rule, removed_leaf in zip(leaf_assignment, removed_leaves):
                        for new_rule in -rule, rule:
                            new_rules = tuple(
                                sorted(removed_leaf.rules + (new_rule,)))

                            # If the new leaf is not already in the cache,
                            # compute its statistics and add it to the
                            # cache
                            if new_rules not in leaf_cache:
                                new_rule_index = abs(new_rule) - 1
                                new_rule_label = int(new_rule > 0)        
                                new_rule_points_cap = rule_vectompz(np.array(x[:, new_rule_index] == new_rule_label) * 1)
                                points_cap, num_captured = rule_vand(removed_leaf.points_cap, new_rule_points_cap)
                                leaf_cache[new_rules] = Leaf(new_rules, points_cap, num_captured, x, y, z, lam)

                            new_leaves.append(leaf_cache[new_rules])

                    child = Tree(x, y, unchanged_leaves + new_leaves, lam, prior_metric)

                    if child in tree_cache:
                        continue

                    if child.loss < best_loss:
                        best_loss = child.loss
                        best_tree = child
                        best_idx = count
                        
                        print(best_loss)
                        print([leaf.rules for leaf in best_tree.leaves])

                    heapq.heappush(queue, (child.loss, child))
                    count += 1
                    if count % 100000 == 0:
                        print("Count: ", count)

    return best_tree
