# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv

##### lambda=0.0035, MAXDEPTH=4, prior_metric="objective",only 4 features (sex:Female, age:18-20, age:21-22, priors:2-3)
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 5.385s | 3.387s | 32,126 | 16,360
With similar support bound | 2701.553s | 553.822s | 21,107 | 13,984
