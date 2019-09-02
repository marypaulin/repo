### Summary of everything I tried

#### Implementation of Similar Support Bound 1: 

For each tree, when we only split one leaf of it, all its child trees are treated as similar trees of each other, and we check the similar support bound for them.

##### use DFS to implement similar support bound (lambda=0.01)

for the simulated data, number of tree evaluations: 37273 (use curiosity for the queue) --> 5971

##### based on DFS, sort the siblings according to their objective value (lambda=0.01)

for the simulated data, number of tree evaluations: 5971 --> 2348

##### based on DFS, check similar support bound the leaf is split by highly correlated features (lambda=0.01)

if any of these features are highly correlated:

| | DFS(similar support when corr>0.975) |	DFS(similar support when corr>0.99)	| DFS(similar support when corr>0.995)	| DFS(similar support when corr>0.999) | DFS(similar support when corr>1)
| ------ | ------ | ------ | ------ | ------ | ------ |
| Time	|	0.549381 | 0.531532 | 0.554438 | 0.570514 | 2.655926
| #Tree Evaluations	| 	5971	| 5971 | 5971 | 5971 | 38756
| #Best Tree |  0 | 0 | 0 | 0 | 6956


if all of these features are highly correlated:


| | DFS(similar support when corr>0) | DFS(similar support when corr>0.8) | DFS(similar support when corr>0.9) | DFS(similar support when corr>0.95) | DFS(similar support when corr>0.975)
| ------ | ------ | ------ | ------ | ------ | ------ |
| Time	|	2.864832 | 2.679301 | 2.629949 | 2.617792 | 2.65786
| #Tree Evaluations	| 	38739 | 38756	| 38756 | 38756 | 38756
| #Best Tree |  6939 | 6956 | 6956 | 6956 | 6956

#### Implementation of Similar Support Bound 2:

For two trees, if we only split one leaf for each of them, if the two leaves capture similar data,
we treat them as similar trees and check the similar support bound.

for the simulated data, only work when Hamming Distance=0, no improment when increase the threshold of the Hamming Distance.
number of tree evaluations: 37273 (use curiosity for the queue) --> 34870

 (lambda=0.01)

#### miscellaneous

The compas data ('./data/compas-binary1.csv') cannot activate the similar support bound.

For the compas data ('./data/compas_from_age_paper.csv') from the age papaer, even when I only use two features (i.e. p_arrest and p_current_age), it is tooooo slow to get the result. These two features correspond to more than 100 binary features.


