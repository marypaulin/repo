# CORELS DT

corels_dt:
all 6907 data from compas-binary.csv

##### lambda=0.0035

##### Try 6 features(sex:Female, age:18-20,age:21-22, juvenile-crimes:=0, priors:2-3, priors:>3)

##### # order x's columns according to gini_reduction
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Last Version. Use the leaf with the minimum loss to find a pair of leaves as d0 (MAXDEPTH = 5) | 2257.721s | 15.745s | 9,844,529 | 341,986
use removed_leaves as d0, unchanged_leaves as dp (MAXDEPTH = 5) | 117.383s | 20.328s | 49,173 | 35,314
use removed_leaves as d0, unchanged_leaves as dp (MAXDEPTH = 6) | 119.512s | 21.726s | 49,215 | 35,344
and check equivalent points bound right after the tree pushed out of the queue (MAXDEPTH = 6) | 5.069s | 2.537s | 53,644 | 35,727
and check accurate support bound for each pair of leaves in new_leaves, rather than check them altogether (MAXDEPTH = 6) | 5.084s | 2.504s | 9,483 | 17,878

##### Try 8 features, MAXDEPTH = 8

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
try 8 features | 30.170s | 5.409s | 435,108 | 40,372

##### Try all 13 features, MAXDEPTH = 13

Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
try all 13 features | 3229.861s | 468.189s | 57,221,882 | 8,559,870