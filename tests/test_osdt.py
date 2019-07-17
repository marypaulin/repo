

import unittest
import numpy as np
from time import time
from multiprocessing import cpu_count

# Test out the cluster library
from lib.data_processing import read_dataset
from lib.parallel_osdt import ParallelOSDT

class TestOSDT(unittest.TestCase):
    pass
    # Consistency test 
    def test_osdt_on_compas(self):
        dataset = read_dataset('data/preprocessed/compas-binary.csv', sep=';')

        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]
        lamb = 0.005

        start = time()
        problem = ParallelOSDT(X, y, lamb, max_time=25)
        print("\nRunning OSDT COMPAS consistency test")
        model = problem.solve(clients=2, servers=1, visualize=True)
        finish = time()
        print('Training Time = {} seconds'.format(round(finish - start, 3)))

        rule_list_visualization = '\n'.join((
        '(_,_,_,0,_,_,_,_,_,_,_,0) => 0 (Risk Contribution = 0.12878746199507746)',
        '(_,_,_,1,_,_,_,0,_,_,_,0) => 1 (Risk Contribution = 0.022518459533806285)',
        '(_,_,_,1,_,_,_,1,_,_,0,0) => 0 (Risk Contribution = 0.0708751990734038)',
        '(_,_,_,1,_,_,_,1,_,_,1,0) => 1 (Risk Contribution = 0.022228898219197917)',
        '(_,_,_,_,_,_,_,_,_,_,_,1) => 1 (Risk Contribution = 0.11155856377587954)'))
        self.assertEqual(model.visualization, rule_list_visualization)
        self.assertEqual(model.risk, 0.35596858259736497)

    # def test_osdt_on_identity(self):
    #     dataset = read_dataset('tests/fixtures/identity.csv', sep=';')

    #     X = dataset.values[:, :-1]
    #     y = dataset.values[:, -1]
    #     lamb = 0.1

    #     problem = ParallelOSDT(X, y, lamb, max_time=1)
    #     print("\nRunning OSDT identity consistency test")
    #     model = problem.solve(clients=2, servers=1, visualize=True)
    #     rule_list_visualization = '\n'.join((
    #         '(_,_,_,_,_,_,_,_,_,_,_,_) => 0 (Risk Contribution = 0.3333333333333333)',))
    #     self.assertEqual(model.visualization, rule_list_visualization)
    #     self.assertEqual(model.risk, 0.3333333333333333)

    # def test_osdt_on_split(self):
    #     dataset = read_dataset('tests/fixtures/split.csv', sep=';')

    #     X = dataset.values[:, :-1]
    #     y = dataset.values[:, -1]
    #     lamb = 0.05

    #     problem = ParallelOSDT(X, y, lamb, max_time=1)
    #     print("\nRunning OSDT split consistency test")
    #     model = problem.solve(clients=2, servers=1, visualize=True)
    #     rule_list_visualization = '\n'.join((
    #     '(_,_,_,_,_,_,_,_,0,_,_,_) => 0 (Risk Contribution = 0.1611111111111111)',
    #     '(_,_,_,_,_,_,_,_,1,_,_,_) => 1 (Risk Contribution = 0.1611111111111111)'))
    #     self.assertEqual(model.visualization, rule_list_visualization)
    #     self.assertEqual(model.risk, 0.3222222222222222)
    
if __name__ == '__main__':
    unittest.main()




