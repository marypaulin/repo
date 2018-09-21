# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv

##### lambda=0.0035, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### This version:
###### MAXDEPTH=4

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 5.075s | 2.883s | 12,764 | 2,927
With similar support bound (add to the head of the deadprefix list) | 12.598s | 3.187s | 9,157 | 2,916

###### MAXDEPTH=5

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 11.018s | 4.440s | 58,505 | 14,826
With similar support bound (add to the head of the deadprefix list) | 319.855s | 16.275s | 48,117 | 14,056



##### last version (Sep 19), lambda=0.0035, MAXDEPTH=4, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3):

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 184.813s | 14.327s | 1,477,303 | 121,295
With similar support bound (add to the head of the deadprefix list) | 739.245s | 103.848s | 264,202 | 119,783