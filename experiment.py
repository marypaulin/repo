# Run some experiments to determine effectiveness of different optimizations on scalability, accuracy, and speed
import math
from sklearn.tree import DecisionTreeClassifier

from lib.osdt_classifier import OSDTClassifier
from lib.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_processing import read_dataset
from lib.analysis import scalability_analysis

# Scalability:
# Fix the configuration
# Measure Process Pool x (N,M) against runtime
# Schema
# dataset, model, n, m, processes, time


# Accuracy:
# Perform cross validation for optimal test accuracy on both Cart, OSDT, and OSDTV3
# Plot for various datasets

# Schema
# dataset, model, configuration, visualization, accuracy (test)


# Measure Speed of Cart, OSDT, OSDTV3 on full-size problems
# Schema
# dataset, model, test_accuracy (cross-validated)

# Benchmark different prioritizations

# datasets = [
#     ('data/preprocessed/compas-binary.csv', ';'),
#     ('data/preprocessed/compas-binary.csv', ';'),
# ]
# models = [
#     (DecisionTreeClassifier,{
#         'max_depth': 5,
#         'min_samples_split': math.ceil(lamb * 2 * fold_size), 
#         'min_samples_leaf': math.ceil(lamb * fold_size), 
#         'max_leaf_nodes': max(2, math.floor(1 / ( 2 * lamb ))), 
#         'min_impurity_decrease': lamb
#     })
#     (OSDTClassifier, { 'regularization': 0.005 })
#     (ParallelOSDTClassifier, { 'regularization': 0.005 })
# ]

regularization = 0.005
dataset_name = 'census'
file_name = 'data/census-data/census_c1s5ky0.2_0_0_test.csv'
dataset = read_dataset(file_name, sep=',')
(n, m) = dataset.shape

model_name = 'cart'
model = DecisionTreeClassifier
hyperparameters = {
        'max_depth': 5,
        'min_samples_split': math.ceil(regularization * 2 * n), 
        'min_samples_leaf': math.ceil(regularization * n), 
        'max_leaf_nodes': max(2, math.floor(1 / ( 2 * regularization ))), 
        'min_impurity_decrease': regularization
}
path = 'data/scalability/{}_{}.csv'.format(model_name, dataset_name)
scalability_analysis(model, hyperparameters, dataset, path)

model_name = 'osdt'
model = OSDTClassifier
hyperparameters = { 'regularization' : regularization }
path = 'data/scalability/{}_{}.csv'.format(model_name, dataset_name)
scalability_analysis(model, hyperparameters, dataset, path)

for core_count in [1, 2, 4, 8, 16, 32, 60]:
    model_name = 'paralle_osdt_{}_core'.format(core_count)
    model = ParallelOSDTClassifier
    hyperparameters = { 'regularization' : regularization, 'clients': core_count }
    path = 'data/scalability/{}_{}.csv'.format(model_name, dataset_name)
    scalability_analysis(model, hyperparameters, dataset, path)