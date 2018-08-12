# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### lambda=0.001, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------
All bounds | 678s | 101s | 7,734,513 | 1,136,376
No support bound | 1369s | 222s | 14,835,715 | 2,733,557
No accurate support bound | 831s | 100s | 8,814,803 | 1,136,376
No equivalent points bound | 710s | 106s | 8,104,054 | 1,189,412
No one-step look ahead bound | 704s | 105s | 7,724,781 | 1,137,218

##### lambda=0.001, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------
All bounds | 318s | 144s | 3,825,748 | 1,743,704
No support bound | 1222s | 449s | 12,059,364 | 4,841,444
No accurate support bound | 401s | 193s | 4,606,894 | 2,205,574
No equivalent points bound | 329s | 157s | 3,986,926 | 1,886,818
No one-step look ahead bound | 324s | 144s | 3,827,758 | 1,744,720

##### lambda=0.001, MAXDEPTH=4, all bounds
metric of the priority queue| total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------
gini | 678s | 101s | 7,734,513 | 1,136,376
entropy | 768s | 115s | 7,658,072 | 1,128,782
curiosity | 1199s | 51s | 16,708,121 | 841,118
lower bound | 1643s | 319s | 21,191,877 | 5,298,510
