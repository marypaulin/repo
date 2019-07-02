import lib.vector as vector
from lib.frequency import count, minority_count

class Leaf:
    population = 0
    cache = {} # This is probably where the LSH table should go for similar support bound

    def __init__(self, rows, columns, y, z, lamb, bounds={}, active=None, rules=None, compress=True):
        self.active = vect.zeros(len(columns)) if active == None else active
        self.rules = vect.zeros(len(columns)) if rules == None else rules
        self.input = (rows, columns, y, z, lamb, bounds, compress)
        self.captures = vect.vectorize([ vect.count(vect.xor(sample, rules) & active) == 0 for sample in rows ])
        self.key = self.captures

    def capture(self, sample):
        return vect.count(vect.xor(sample, self.rules) & self.active) == 0

    def predict(self, sample):
        return self.query('prediction') if self.capture(sample) else None

    def query(self, attribute):
        if self.input != None:
            self.__load__()
        return self.state[attribute]

    def key(self):
        # return (self.active, self.rules)
        return self.key

    def children(self, rows, columns, y, z, lamb, bounds={}, compress=True):
        # TODO:
        # Return a list of tuples containg pairs of valid initializers for new leaves
        # Empty if the leaf is non-split
        for j in (j for j in len(self.active) if not vect.test(self.active, j)):
            if 

        return

    def __load__(self):
        key = self.key()
        if cache.get(key) != None:
            cache[key] = self.__compute_state__()
            population += 1
        self.state = cache[key]
        self.input = None
    
    def __compute_state__(self):
        (rows, columns, y, z, lamb, bounds, compress) = self.input
        active = self.active
        rules = self.rules
        # n-bit capture vector of this leaf of the n rows of X
        captures = self.captures

        # Compute various frequencies
        if self.compress:
            total_frequency = count(z)
            capture_frequency = count(z, indicator=captures)
            minority_frequency = minority_count(z, indicator=captures)
            label_1_frequency = count(z, indicator=captures, label=1)
            
        else:
            total_frequency = len(y)
            capture_frequency = vect.count(captures)
            minority_frequency = vect.count(captures & z)
            label_1_frequency = vect.count(captures & y)

        # b_0: unavoidable misclassification (Equivalents Points Bound / Theorem 9)
        b_0 = minority_frequency / total_frequency
        
        # normalized support of the leaf (Lower Bound on Leaf Support of Leaf Pairs / Theorem 3)
        support = capture_frequency / total_frequency 
        split = support >= 2 * lamb

        # proportion of captures samples labeled 1
        proportion_1 = label_1_frequency / capture_frequency
        # majority prediction
        prediction = int(proportion_1 >= 0.5)
        # proportion of samples mispredicted (over all n samples)
        loss = min(proportion_1, 1-proportion_1) * capture_frequency / n

        # Otherwise the leaf is already terminal and therefore the misclassification is fixed to the current loss
        if split:
            # If the leaf can be split then the misclassification can only be known to a lowerbound using Theorem 9
            lowerbound = b_0
        else:
            # If the leaf cannot be split then the misclassification is known exactly as loss
            lowerbound = loss

        # Apply Theorem 5 using loss instead of accuracy.
        # Conversion: accuracy = support - loss
        fails_leaf_accuracy_lowerbound = support - lamb < lowerbound

        # Apply Theorem 3.
        fails_leaf_support_lowerbound = not split

        # Combining with Theorem 3:
        # Any tree containing this leaf cannot possibly be optimal or have a child that is optimal
        # Intuittion: The leaf fails the accuracy lower bound, and accuracy does not improve in this leaf
        # because its support is too low to split
        prune = fails_leaf_accuracy_lowerbound and fails_leaf_support_lowerbound

        return {
            'split': split, # Indicates whether the leaf can be split
            'prune': prune, # Indicates that the tree containing this leaf can provably be pruned
            'prediction': prediction, # majority label of this leaf
            'loss': loss, # normalized misclassification of this leaf
            'lowerbound': lowerbound # lowerbound on misclassification of this leaf and any further splits
        }
