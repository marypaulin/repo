# CORELS DT

##### compas-binary.csv, 13 features, 6907 data
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue | optimal tree size
  ------------- | ------------- | ------------- | -------------  | -------------  | -------------
lambda=0.0035 | 3365.399s | 474.551s | 57,221,882 | 8,559,870 | 6
lambda=0.005 | 542.960s | 33.879s | 10,337,041 | 530,556 | 3
lambda=0.05 | 0.369s | 0.038s | 26 | 1 | 2


#### [MONK's Problems dataset](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)
#### [MONK's Problems](https://www.researchgate.net/profile/Yoram_Reich/publication/2293492_The_MONK's_Problems_A_Performance_Comparison_of_Different_Learning_Algorithms/links/57358d6208ae9f741b2987fb/The-MONKs-Problems-A-Performance-Comparison-of-Different-Learning-Algorithms.pdf)

##### monk1-train.csv, 11 features, 124 data
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue | optimal tree size
  ------------- | ------------- | ------------- | -------------  | -------------  | -------------
lambda=0.05 | 0.368s | 0.103s | 4,748 | 749 | 5
lambda=0.02 | 47.817s | 0.107s | 1,158,145 | 687 | 8

##### monk2-train.csv, 11 features, 169 data
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue | optimal tree size
  ------------- | ------------- | ------------- | -------------  | -------------  | -------------
lambda=0.035 | 880.414s | 0s | 4,291,599 | 0 | 0


##### monk3-train.csv, 11 features, 169 data
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue | optimal tree size
  ------------- | ------------- | ------------- | -------------  | -------------  | -------------
lambda=0.02 | 5542.682s | 10.326s | 70,428,060 | 68,021 | 9
lambda=0.03 | 700.427s | 0.330s | 9,121,614 | 2,921 | 5


#### [Voting Records dataset](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records)
##### voting-records-binary.csv, 16 features, 232 data
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue | optimal tree size
  ------------- | ------------- | ------------- | -------------  | -------------  | -------------
lambda=0.0035 | >10h |  |  |  | 