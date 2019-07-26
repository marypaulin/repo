# third-party imports
from time import time

# local imports
from lib.models.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_structures.dataset import read_dataframe

# Using COMPAS as an example
dataset = read_dataframe('data/preprocessed/compas-binary.csv')
(n, m) = dataset.shape
X = dataset.values[:n, :m-1]
y = dataset.values[:n, -1]

hyperparameters = {
    # Regularization coefficient which effects the penalty on model complexity
    'regularization': 0.005,

    'max_depth': float('Inf'),  # User-specified limit on the model
    'max_time': float('Inf'),  # User-specified limit on the runtime

    'workers': 1,  # Parameter that varies based on how much computational resource is available

    'profile': True, # Toggle Snapshots for Profiling Memory Usage

}

start = time()
model = ParallelOSDTClassifier(**hyperparameters)
model.fit(X, y)
print('Runtime: {} Seconds'.format(time() - start))
print('Prediction: \n{}'.format(model.predict(X)))
print('Training Accuracy: {}'.format(model.score(X, y)))
print('Visualization: \n{}'.format(model.model.visualization))
