# Run some experiments to determine effectiveness of different optimizations on scalability, accuracy, and speed
import sys
from math import ceil, floor
from os import mkdir
from os.path import basename, splitext, exists
from gc import collect
from time import sleep
from sklearn.tree import DecisionTreeClassifier

from lib.models.osdt_classifier import OSDTClassifier
from lib.models.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_structures.dataset import read_dataframe
from lib.experiments.analysis import scalability_analysis

# Extract Arguments
# arguments: <dataset_path> <timeout?> <regularization?> <model_name?> <core_count?>

arguments = sys.argv
input_path = arguments[1]
basename = basename(input_path)
dataset_name, extension = splitext(basename)
if not exists('data/scalability/{}'.format(dataset_name)):
    mkdir('data/scalability/{}'.format(dataset_name))
dataset = read_dataframe(input_path)
(n, m) = dataset.shape

timeout = float(arguments[2]) if len(arguments) >= 3 else 60
regularization = float(arguments[3]) if len(arguments) >= 4 else 0.1
model_name = arguments[4] if len(arguments) >= 5 else 'osdt'
core_count = int(arguments[5]) if len(arguments) >= 6 else 1


# Display Configurations
print("Running Scalability Data Collection")
print("Dataset: {}".format(input_path))
print("Regularization: {}".format(regularization))
print("Timeout: {}".format(timeout))
print("Model: {}".format(model_name))
print("Core Count: {}".format(core_count))
sleep(3)

if model_name == 'cart':
    # Run Data Collection for CART
    model = DecisionTreeClassifier
    hyperparameters = {
            'max_depth': 5,
            'min_samples_split': ceil(regularization * 2 * n), 
            'min_samples_leaf': ceil(regularization * n), 
            'max_leaf_nodes': max(2, floor(1 / ( 2 * regularization ))), 
            'min_impurity_decrease': regularization
    }
    output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
    scalability_analysis(dataset, model, hyperparameters, output_path)

elif model_name == 'osdt':
    # Run Data Collection for OSDT
    model = OSDTClassifier
    hyperparameters = {'regularization': regularization, 'max_time': timeout}
    output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
    scalability_analysis(dataset, model, hyperparameters, output_path)

elif model_name == 'parallel_osdt':
    # Run Data Collection for Parallel OSDT
    model_name = 'parallel_osdt_{}_core'.format(core_count)
    model = ParallelOSDTClassifier
    hyperparameters = { 'regularization' : regularization, 'workers': core_count, 'max_time': timeout }
    output_path = 'data/scalability/{}/{}.csv'.format(dataset_name, model_name)
    scalability_analysis(dataset, model, hyperparameters, output_path)
