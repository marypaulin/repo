# third-party imports
import pandas as pd
from sklearn.utils import shuffle

# Overview: Module for handling data processing

# Summary: Read in the datasets and returns a Pandas dataframe
# Input:
#   path: relative path to csv path
#   sep: separation character of csv
# Output:
#   dataset: a Pandas dataframe containing the dataset at given path
def read_dataset(path, sep=',', randomize=False):
    dataset = pd.DataFrame(pd.read_csv(path, sep=sep))
    if randomize:
        dataset = shuffle(dataset)
    return dataset

