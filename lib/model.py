# third-party imports
import time
import numpy as np

# local imports
from lib.osdt import bbound
from lib.logger import Logger
import lib.vector as vect
from lib.frequency import count
from lib.leaf import Leaf
from lib.tree import Tree
from lib.queue import Queue

class OSDT_OLD:
    """
    Model Interface for external interaction
    """
    def __init__(self, 
                regularizer = 0.1, # lambda but Python doesn't let us use lambda
                priority_metric = 'curiosity',
                max_depth = None,
                max_width = None,
                max_time = None,
                compress = True,
                bounds = None,
                execution = None,
                verbose = True,
                log = True):

        self.regularizer = regularizer
        self.priority_metric = 'curiosity'

        self.max_depth = max_depth
        self.max_width = max_width
        self.max_time = max_time

        self.bounds = bounds if bounds != None else self.__default_bounds__()
        self.execution = execution if execution != None else self.__default_execution__()
        self.verbose = verbose
        self.logger = Logger() if log else None

        # This makes some memory savings and *possibly* time savings depending on compression rate
        self.compress = compress # Experimental feature that operates on compressed data

    def fit(self, X, y):

        # predictor = bbound(training_features, training_labels, self.regularizer,
        #     prior_metric='curiosity', MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
        #     support=True, incre_support=True, accu_support=True, equiv_points=True,
        #     lookahead=True, lenbound=True, R_c0=1, timelimit=float('Inf'), init_cart=True,
        #     saveTree=False, readTree=False)

        self.__fit__(X, y)
        return

    def predict(self, X_hat):
        if self.__predictor__ == None:
            raise "Error: Model not yet trained"
        
        prediction, _accuracy = self.__predictor__(X_hat)
        return prediction

    def score(self, X_hat, y_hat):
        if self.__predictor__ == None:
            raise "Error: Model not yet trained"
        _prediction, accuracy = self.__predictor__(X_hat, labels=y_hat)
        self.accuracy = accuracy
        return accuracy

    def __default_bounds__(self):
        return {
            'incremental_support': True,
            'minimum_support': True,
            'minimum_accuracy': True,
            'equivalent_points': True,
            'look-ahead': True,
            'length': True,
        }
    
    def __default_execution__(self):
        return {

        }
    
    def __time__(self):
        return time.time() - self.start_time

    def __print__(self, message):
        # Internal print method to deal with verbosity and logging
        if self.verbose:
            print(message)
        if self.logger != None:
            self.logger.log([time.time(), message])


    def __fit__(self, X, y):
        # n = number of rows (samples)
        # m = number of columns (features)
        (n, m) = X.shape
        self.__print__("Training Data Dimensions: {} samples, {} features".format(n, m))
        self.start_time = time.time()
        self.max_width = 2 ** m if self.max_width == None else self.max_width

        # Reduce the training data using equivalent points
        z, X_rows, X_columns = self.__summarize__(X, y)
        y = vect.vectorize(y)

        # Reorder columns by descending gini coefficient of resulting splits
        gini_reduction_index = self.__gini_reduction_index__(X_rows, X_columns, y, z)

        # Initial state variables
        initial_tree = Tree(X_rows, X_columns, y, z, lamb, priority_metric='curiosity', bounds=self.bounds, compress=self.compress))
        queue = Queue(init=[Task(lambda : self.__explore__(initial_tree))])

        root_leaf = CacheLeaf(ndata, (), y_mpz, z_mpz, make_all_ones(
            ndata + 1), ndata, lamb, support, [0] * nrule)

        d_c = CacheTree(leaves=[root_leaf], lamb=lamb)
        R_c = d_c.risk
        C_c = 0
        time_c = time.time()


        # tree = Tree(cache_tree=d_c, lamb=lamb,
        #             ndata=ndata, splitleaf=[1], prior_metric=prior_metric)
                    
        # queue.push(Tree())

        # while queue.size() > 0 and self.__time__() < self.max_time:
    
    def __explore__(self, tree):
        

    def __summarize__(self, X, y):
        """
        Summarize the training dataset via equivalent points
        This simultaneously computes:
         - equivalent points bound data
         - majority labels for leaf construction,
        also reduce the row count which makes column vectors smaller

        Return values:
        z is a dictionary in which:
         - keys are unique combinations of features which represent an equivalent point set
         - values are tuples representing the frequencies of labels in that equivalent point set

        rows is the bit-vector rows filtered down to only unique ones
        columns is the bit-vector columns shortened to only the unique rows
        """
        (n, m) = X.shape
        # Vectorize features and labels for bitvector operations
        columns = [vect.vectorize(X[:, j]) for j in range(m)]
        rows = [vect.vectorize(X[i, :]) for i in range(n)]
        
        z = {}
        for i in range(n):
            row = rows[i]
            if z.get(row) == None:
                # z stores a tuple for each unique row (equivalent point set)
                # the tuple stores (in order) the frequency of labels 0 and 1
                z[row] = [0, 0]
            
            # Increment the corresponding label frequency in the equivalent point set
            z[row][y[i]] += 1
        
        reduced_rows = list(z.keys())

        compression_rate = len(rows) / len(reduced_rows)
        self.__print__("Row Compression Rate: {}".format(compression_rate))

        # This causes column vectors to have lots of leading zeros (except the first one)
        # it may help later for doing more compression
        list.sort(reduced_rows)
        list.reverse(reduced_rows)

        z = [ z[row] for row in reduced_rows ]
        reduced_columns = [ vect.vectorize([ vect.read(row, j) for row in reduced_rows]) for j in range(m) ]

        if self.compress:
            return z, reduced_rows, reduced_columns
        else:
            return z, rows, columns

    def __reduction__(self, captures, x_j, y, z):
        """
        computes the weighted sum of Gini coefficients of bisection subsets by feature j
        reference: https://en.wikipedia.org/wiki/Gini_coefficient
        """
        if self.compress:
            negative_captures = captures & vect.negate(x_j)
            negative_frequency = count(z, indicator=negative_captures)
            p_1 = count(z, indicator=negative_captures, label=1) / negative_frequency if negative_frequency > 0 else 0

            positive_captures = captures & x_j
            positive_frequency = count(z, indicator=positive_captures)
            p_2 = count(z, indicator=positive_captures, label=1) / positive_frequency if positive_frequency > 0 else 0
        else:
            negative = captures & vect.negate(x_j)
            negative_frequency = vect.count(negative)
            # negative correlation of feature j with labels
            p_1 = vect.count(negative & y) / negative_frequency if negative_frequency > 0 else 0

            positive = captures & x_j
            positive_frequency = vect.count(positive)
            # positive correlation of feature j with labels
            p_2 = vect.count(positive & y) / positive_frequency if positive_frequency > 0 else 0

        gini_1 = 2 * p_1 * (1 - p_1) # Degree of inequality of labels in samples of negative feature
        gini_2 = 2 * p_2 * (1 - p_2) # Degree of inequality of labels in samples of positive feature
        # Base inequality minus inequality 
        reduction = negative_frequency * gini_1 + positive_frequency * gini_2
        return reduction

    def __gini_reduction_index__(self, rows, columns, y, z, captures=None):
        """
        calculate the gini reduction by each feature
        return the rank of by descending
        """
        (n, m) = (len(rows), len(columns))
        captures = vect.ones(n) if captures == None else captures
        # Sets the subset we compute probability over
        if self.compress:
            capture_frequency = count(z, indicator=captures)
            p_0 = count(z, indicator=captures, label=1) / capture_frequency if capture_frequency > 0 else 0
        else:
            capture_frequency = vect.count(captures)
            p_0 = vect.count(y & captures) / capture_frequency if capture_frequency > 0 else 0

        gini_0 = 2 * p_0 * (1 - p_0)

        # The change in degree of inequality when splitting the capture set by each feature j
        # In general, positive values are similar to information gain while negative values are similar to information loss
        # All values are negated so that the list can then be sorted by descending information gain
        reductions = np.array([ -(gini_0 - self.__reduction__(captures, column, y, z) / capture_frequency) for column in columns])
        order_index = reductions.argsort()

        self.__print__("Negative Gini Reductions: " + str(reductions))
        self.__print__("Negative Gini Reductions Index: " + str(order_index))
 
        return order_index
