# Run some experiments to determine effectiveness of different optimizations on scalability, accuracy, and speed
import math
import sys
import os
from time import sleep
from sklearn.tree import DecisionTreeClassifier

from lib.osdt_classifier import OSDTClassifier
from lib.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_processing import read_dataset
from lib.analysis import scalability_analysis

# Extract Arguments
arguments = sys.argv
input_path = arguments[1]
basename = os.path.basename(input_path)
dataset_name, extension = os.path.splitext(basename)
if not os.path.exists('data/scalability/{}'.format(dataset_name)):
    os.mkdir('data/scalability/{}'.format(dataset_name))
dataset = read_dataset(input_path)
(n, m) = dataset.shape

timeout = float(arguments[2]) if len(arguments) >= 3 else 60
regularization = float(arguments[3]) if len(arguments) >= 4 else 0.1

# Display Configurations
print("Running Scalability Data Collection")
print("Dataset: {}".format(input_path))
print("Regularization: {}".format(regularization))
print("Timeout: {}".format(timeout))
sleep(3)

# Run Data Collection for CART
model_name = 'cart'
model = DecisionTreeClassifier
hyperparameters = {
        'max_depth': 5,
        'min_samples_split': math.ceil(regularization * 2 * n), 
        'min_samples_leaf': math.ceil(regularization * n), 
        'max_leaf_nodes': max(2, math.floor(1 / ( 2 * regularization ))), 
        'min_impurity_decrease': regularization
}
output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
scalability_analysis(dataset, model, hyperparameters, output_path)

# Run Data Collection for OSDT
model_name = 'osdt'
model = OSDTClassifier
hyperparameters = {'regularization': regularization, 'max_time': timeout}
output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
scalability_analysis(dataset, model, hyperparameters, output_path)

# Run Data Collection for Parallel OSDT
for core_count in [1, 2, 4, 8, 16, 32, 60]:
    model_name = 'parallel_osdt_{}_core'.format(core_count)
    model = ParallelOSDTClassifier
    hyperparameters = { 'regularization' : regularization, 'clients': core_count, 'max_time': timeout }
    output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
    scalability_analysis(dataset, model, hyperparameters, output_path)
