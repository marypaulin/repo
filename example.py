# third-party imports
import cProfile
from time import time
import sys
import pickle

# local imports
from lib.models.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.models.osdt_metrics_classifier import OSDTMetricsClassifier
from lib.data_structures.dataset import read_dataframe

# Using COMPAS as an example

dataset = read_dataframe('data/preprocessed/fico_binary.csv')
# dataset = read_dataframe('data/preprocessed/compas-binary.csv')
# dataset = read_dataframe('data/preprocessed/small_example.csv')
regularization = 0.005

# dataset = read_dataframe('data/preprocessed/census.csv')
# regularization = 0.2

(n, m) = dataset.shape
workers = 1

# arguments: <processes> <subsample?> <subfeature?>
if len(sys.argv) >= 2:
    workers = int(sys.argv[1])
if len(sys.argv) >= 3:
    n = int(sys.argv[2])
if len(sys.argv) >= 4:
    m = int(sys.argv[3])
if len(sys.argv) >= 5:
    regularization = float(sys.argv[4])

print('Regularization = {}, N = {}, M = {}, workers = {}'.format(regularization, n, m, workers))

profile = False

X = dataset.values[:n, :m-1]
y = dataset.values[:n, -1]

hyperparameters = {
    # Regularization coefficient which effects the penalty on model complexity
    'regularization': regularization,

    'max_depth': float('Inf'),  # User-specified limit on the model
    'max_time': float('Inf'),  # User-specified limit on the runtime

    #'workers': workers,  # Parameter that varies based on how much computational resource is available

    #'visualize_model': True,  # Toggle whether a rule-list visualization is rendered
    #'visualize_training': False,  # Toggle whether a dependency graph is streamed at runtime
    'verbose': False,  # Toggle whether event messages are printed
    'log': False,  # Toggle whether client processes log to logs/work_<id>.log files
    #'profile': False,  # Toggle Snapshots for Profiling Memory Usage

    'configuration': {  # More configurations around toggling optimizations and prioritization options
        # 'priority_metric': ['depth', 'lowerbound', 'support'],  # Decides how tasks are prioritized
        'priority_metric': ['depth'],
        # Decides how much to push back a task if it has pending dependencies
        'deprioritization': 0.1,
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
        'equivalent_point_compression': False,
        # Toggles whether asynchronous tasks can be cancelled after being issued
        'task_cancellation': False,
        # Toggles whether look_ahead prunes using objective upperbounds (This builds on top of look_ahead)
        'interval_look_ahead': True,
        # Cooldown timer (seconds) on synchornization operations
        'synchronization_cooldown': 0.1,
        # Probability of saying "Fine. I will do it myself."
        'independence': 0
    }
}

start = time()
model = ParallelOSDTClassifier(**hyperparameters)
# model = OSDTMetricsClassifier(**hyperparameters)

if __name__ == "__main__":
    if profile:
        cProfile.run('model.fit(X, y)', sort='tottime')
    else:
        model.fit(X, y)
    prediction = model.predict(X)
    prediction = prediction.reshape(1, n)[0]
    print('Runtime: {} Seconds'.format(time() - start))
    print('Prediction: \n{}'.format(prediction))
    print('Training Accuracy: {}'.format(model.score(X, y)))
    # print('Visualization: \n{}'.format(model.model.visualization))

    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.load(open('model.pkl', 'rb'))
