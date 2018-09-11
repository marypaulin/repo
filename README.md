# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv

### calculate similar support bound when the highly correlated features substitue each other; regular expression is used, and the preformace is SOOOOOO bad, SOOOOOO slow

##### lambda=0.0035, MAXDEPTH=4, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound |  |  |  | 
With similar support bound |  |  |  | 

##### lambda=0.0035, MAXDEPTH=4, prior_metric="objective",only 4 features (sex:Female, age:18-20, age:21-22, priors:2-3)
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound |  |  |  | 
With similar support bound |  |  |  | 
