# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv

##### lambda=0.0035, 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)
##### MAXDEPTH = 5, Without similar support bound

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
One copy for each tree (prior_metric="objective") | 6.171s | 3.385s | 18,855 | 5,525
Multiple copies for each tree (prior_metric="objective") | 4.960s | 3.650s | 47,780 | 26,561
One copy for each tree (prior_metric="bound") | 6.355s | 4.699s | 17,714 | 5,598
Multiple copies for each tree (prior_metric="bound") | 31.165s | 3.397s | 773,889 | 3,862
One copy for each tree (prior_metric="curiosity") | 5.796s | 5.222s | 10,672 | 7,136
Multiple copies for each tree (prior_metric="curiosity") | 6.391s | 5.025s | 86,712 | 46,334
One copy for each tree (prior_metric="gini") | 6.170s | 2.762s | 13,679 | 1,059
Multiple copies for each tree (prior_metric="gini") | 7.997s | 0.769s | 112,619 | 44


##### lambda=0.0035, 6 features: 5 features+1 manually added highly correlated feature of age:21-22 (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)
##### MAXDEPTH = 4, One copy, prior_metric="objective"

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 15.236s | 7.316s | 71,282 | 23,402
With SSB when sub v2 | 14.871s | 7.335s | 66,473 | 21,948



##### lambda=0.0035, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### The version (Oct 2) with tighter incremental support bound (#corr/ndata <= lambda):

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound (MAXDEPTH=4)| 4.477s | 2.795s | 6,864 | 1,794
With similar support bound (MAXDEPTH=4) | 7.858s | 3.014s | 6,161 | 1,794
With SSB when big (MAXDEPTH=4, SSB_cache_thres=5, SSB_check_thres=6) | 4.631s | 2.767s | 6,618 | 1,794
With SSB using priority queue ordered by #similarity (MAXDEPTH=4) | 8.540s | 3.027s | 6,161 | 1,794
Without similar support bound (MAXDEPTH=5)  | 6.143s | 3.407s | 18,855 | 5,525
With similar support bound (MAXDEPTH=5) | 54.607s | 5.776s | 17,633 | 5,525
With SSB when big (MAXDEPTH=5, SSB_cache_thres=5, SSB_check_thres=6) | 7.613s | 3.545s | 18,693 | 5,255
With SSB using priority queue ordered by #similarity (MAXDEPTH=5) | 65.060s | 6.389s | 17,633 | 5,525

##### try 6 features, lambda = 0.01, MAXDEPTH=6

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound (incremental support bound)  | 206.155s | 0.498s | 1,336,065 | 11
Without similar support bound (tighter incremental support bound)  | 11.318s | 0.498s | 20,382 | 11

##### lambda=0.0035, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
With tighter incremental support bound (5 features, MAXDEPTH=4)| 4.477s | 2.795s | 6,864 | 1,794
Without tighter incremental support bound (5 features, MAXDEPTH=4) | 184.813s | 14.327s | 1,477,303 | 121,295



##### This version with incremental support bound (support <= lambda):
###### MAXDEPTH=4

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound  (5 features, MAXDEPTH=4)| 5.075s | 2.883s | 12,764 | 2,927
With similar support bound (5 features, MAXDEPTH=4)| 12.598s | 3.187s | 9,157 | 2,916
Without similar support bound  (5 features, MAXDEPTH=5)| 11.018s | 4.440s | 58,505 | 14,826
With similar support bound (5 features, MAXDEPTH=5)| 319.855s | 16.275s | 48,117 | 14,056



##### last version (Sep 19), lambda=0.0035, MAXDEPTH=4, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3):

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 184.813s | 14.327s | 1,477,303 | 121,295
With similar support bound (add to the head of the deadprefix list) | 739.245s | 103.848s | 264,202 | 119,783