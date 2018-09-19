# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv

##### lambda=0.0035, MAXDEPTH=4, prior_metric="objective",5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### This version:


###### use leaves' actual loss for support bound and accurate support bound
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 184.813s | 14.327s | 1,477,303 | 121,295
With similar support bound (add to the head of the deadprefix list) | 739.245s | 103.848s | 264,202 | 119,783


Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 249.140s | 62.677s | 2,053,987 | 499,510
With similar support bound (append to the end of the deadprefix list) | 1119.532s | 387.051s | 471,927 | 299,831
With similar support bound (add to the head of the deadprefix list) | 824.534s | 286.960s | 471,927 | 299,831


##### last version (Sep 14):

Algorithm variant (last version, Sep 14) | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Without similar support bound | 274.843s | 68.281s | 2,053,987 | 499,510
With similar support bound (activate when highly correlated features substitute each other; use regular expression) | >10h |  |  | 
With similar support bound (append to the end of the deadprefix list) | 7458.143s | 3330.492s | 395,261 | 281,332
With similar support bound (append to the end of the deadprefix list; activate when #leaves>4) | 9282.356s | 3656.371s | 418,928 | 296,389
With similar support bound (add to the head of the deadprefix list) | 6817.165s | 2978.337s | 395,261 | 281,332
With similar support bound (drop the element in deadprefix list when it is just used) | >4h |  |  | 
With similar support bound (use priority queue to cache deadprefix, the metric is the objective, i.e. the same as the queue of trees) | 13787.366s | 6265.261s | 395,261 | 281,332
