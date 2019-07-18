# third-party imports
import cProfile
from time import time
import sys

# local imports
from lib.models.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_structures.dataset import read_dataframe

# Using COMPAS as an example
dataset = read_dataframe('data/preprocessed/compas-binary.csv')
(n, m) = dataset.shape

# arguments: <processes> <subsample?> <subfeature?>
if len(sys.argv) >= 3:
    n = int(sys.argv[2])
if len(sys.argv) >= 4:
    m = int(sys.argv[3])


X = dataset.values[:n, :m-1]
y = dataset.values[:n, -1]

hyperparameters = {
    # Regularization coefficient which effects the penalty on model complexity
    'regularization': 0.005,

    'max_depth': float('Inf'),  # User-specified limit on the model
    'max_time': 60,  # User-specified limit on the runtime

    'workers': int(sys.argv[1]),  # Parameter that varies based on how much computational resource is available

    'visualize': True,  # Toggle whether a rule-list visualization is rendered
    'verbose': True,  # Toggle whether event messages are printed
    'log': False,  # Toggle whether client processes log to logs/work_<id>.log files
    'profile': False,  # Toggle Snapshots for Profiling Memory Usage

    'configuration': {  # More configurations around toggling optimizations and prioritization options
        'priority_metric': 'uniform',  # Decides how tasks are prioritized
        # Decides how much to push back a task if it has pending dependencies
        'deprioritization': 0.01,

        # Note that Leaf Permutation Bound (Theorem 6) is
        # Toggles the assumption about objective independence when composing subtrees (Theorem 1)
        # Disabling this actually breaks convergence due to information loss
        'hierarchical_lowerbound': True,
        # Toggles whether problems are pruned based on insufficient accuracy (compared to other results) (Lemma 2)
        'look_ahead': True,
        # Toggles whether a split is avoided based on insufficient support (proxy for accuracy gain) (Theorem 3)
        'support_lowerbound': True,
        # Toggles whether a split is avoided based on insufficient potential accuracy gain (Theorem 4)
        'incremental_accuracy_lowerbound': True,
        # Toggles whether a problem is pruned based on insufficient accuracy (in general) (Theorem 5)
        'accuracy_lowerbound': True,
        # Toggles whether problem equivalence is based solely on the capture set (Similar to Corollary 6)
        'capture_equivalence': True,
        # Hamming distance used to propagate bounding information of similar problems (Theorem 7 + some more...)
        "similarity_threshold": 0,
        # Toggles whether equivalent points contribute to the lowerbound (Proposition 8 and Theorem 9)
        'equivalent_point_lowerbound': True,

        # Toggles compression of dataset based on equivalent point aggregation
        'equivalent_point_compression': True,
        # Toggles whether asynchronous tasks can be cancelled after being issued
        'task_cancellation': True,
        # Toggles whether look_ahead prunes using objective upperbounds (This builds on top of look_ahead)
        'interval_look_ahead': True,
        # Cooldown timer (seconds) on synchornization operations
        'synchronization_cooldown': 0.01,
        # Cache Limit
        'cache_limit': float('Inf')
    }
}

start = time()
model = ParallelOSDTClassifier(**hyperparameters)
cProfile.run('model.fit(X, y)', sort='cumtime')
# model.fit(X, y)
prediction = model.predict(X)
prediction = prediction.reshape(1, n)
print('Runtime: {} Seconds'.format(time() - start))
print('Prediction: \n{}'.format(prediction))
print('Training Accuracy: {}'.format(model.score(X, y)))
print('Visualization: \n{}'.format(model.model.visualization))
