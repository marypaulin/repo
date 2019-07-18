import unittest

from lib.data_structures.dataset import DataSet, read_dataframe

class TestDataset(unittest.TestCase):

    def test_preprocessing(self):
        # Identity is a dataset where the features form a 12x12 identity matrix, stacked vertically 3 times
        # The labels are distributed so that each equivalent group has two 0's and one 1
        dataframe =  read_dataframe('tests/fixtures/identity.csv', sep=';', randomize=True)
        # dataframe =  read_dataset('data/preprocessed/compas-binary.csv', sep=';', randomize=True)

        X = dataframe.values[:, :-1]
        y = dataframe.values[:, -1]
        dataset = DataSet(X, y)

        # Dataset guarantees that gini ranking is an all-way tie
        # This forces the gini index to maintain the original ranking
        # This should hold true no matter how much we shuffle the rows before hand
        gini_index = dataset.gini_index
        self.assertEqual(dataset.compression_rate, 3.0)
        self.assertEqual(gini_index, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))

    def test_label_distribution(self):
        # Identity is a dataset where the features form a 12x12 identity matrix, stacked vertically 3 times
        # The labels are distributed so that each equivalent group has two 0's and one 1
        dataframe = read_dataframe('tests/fixtures/identity.csv', sep=';', randomize=True)

        X = dataframe.values[:, :-1]
        y = dataframe.values[:, -1]
        dataset = DataSet(X, y)

        label_distribution = dataset.label_distribution()

        self.assertEqual(tuple(label_distribution), (36, 24, 12, 12, 24))

    def test_splits(self):
        # Identity is a dataset where the features form a 12x12 identity matrix, stacked vertically 3 times
        # The labels are distributed so that each equivalent group has two 0's and one 1
        dataframe = read_dataframe('tests/fixtures/identity.csv', sep=';', randomize=True)

        X = dataframe.values[:, :-1]
        y = dataframe.values[:, -1]
        dataset = DataSet(X, y)
        
        # Each group unqiuely sets one feature so each feature positively select 1 group and negatively selects 11 groups
        # 3 samples per group, a root split has 3 in-group and 33 out-group
        for j, left, right in dataset.splits():
            self.assertEqual(left.count(), 11)
            self.assertEqual(right.count(), 1)
            # Out-group contains 33 rows, with 2:1 zero-one ratio
            self.assertEqual(tuple(dataset.label_distribution(left)), (33, 22, 11, 11, 22))
            # In-group contains 3 rows, with 2:1 zero-one ratio
            self.assertEqual(tuple(dataset.label_distribution(right)), (3, 2, 1, 1, 2))

if __name__ == '__main__':
    unittest.main()
