
import math
import sys
import os
from time import sleep
from sklearn.tree import DecisionTreeClassifier

from lib.osdt_classifier import OSDTClassifier
from lib.parallel_osdt_classifier import ParallelOSDTClassifier
from lib.data_processing import read_dataset
from lib.analysis import accuracy_analysis

# Extract Arguments
arguments = sys.argv
input_path = arguments[1]
basename = os.path.basename(input_path)
dataset_name, extension = os.path.splitext(basename)
if not os.path.exists('data/accuracy/{}'.format(dataset_name)):
    os.mkdir('data/accuracy/{}'.format(dataset_name))
dataset = read_dataset(input_path)
(n, m) = dataset.shape

timeout = float(arguments[2]) if len(arguments) >= 3 else 60

# Display Configurations
print("Running Accuracy Data Collection")
print("Dataset: {}".format(input_path))
print("Timeout: {}".format(timeout))
sleep(3)

regularizations = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]

model_name = 'cart'
model = DecisionTreeClassifier
hyperparameters = [{
    'max_depth': None,
    'min_samples_split': math.ceil(regularization * 2 * n),
    'min_samples_leaf': math.ceil(regularization * n),
    'max_leaf_nodes': max(2, math.floor(1 / (2 * regularization))),
    'min_impurity_decrease': regularization
} for regularization in regularizations]
output_path = 'data/accuracy/{}/{}.csv'.format(dataset_name, model_name)
accuracy_analysis(dataset, model, hyperparameters, output_path)

model_name = 'osdt'
model = OSDTClassifier
hyperparameters = [{'regularization': regularization, 'max_time': timeout} for regularization in regularizations]
output_path = 'data/accuracy/{}/{}.csv'.format(dataset_name, model_name)
accuracy_analysis(dataset, model, hyperparameters, output_path)

# Run Data Collection for OSDT
model_name = 'parallel_osdt'
model = ParallelOSDTClassifier
hyperparameters = [{'regularization': regularization, 'visualize':True, 'max_time': timeout} for regularization in regularizations]
output_path = 'data/accuracy/{}/{}.csv'.format(dataset_name, model_name)
accuracy_analysis(dataset, model, hyperparameters, output_path)
