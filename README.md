# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

#### lambda=0.001, MAXDEPTH=4, prior_metric="objective"

##### this version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 580.179s | 2.302s | 4,285,056 | 1,370
No accurate support bound | 587.800s | 2.309s | 4,285,056 | 1,370
No equivalent points bound | 896.611s | 2.299s | 5,967,395 | 1,370
No one-step look ahead bound | 597.133s | 2.233s | 4,408,101 | 1,370

##### the old version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 586.285s | 1.753s | 6,307,044 | 672
No accurate support bound | 691.745s | 1.711s | 6,435,730 | 672
No equivalent points bound | 612.924s | 2.307s | 6,340,464 | 1,354
No one-step look ahead bound | 584.871s | 1.741s | 6,301,576 | 672


#### lambda=0.0025, MAXDEPTH=4, prior_metric="objective"

##### this version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 327.208s | 60.741s | 2,470,538 | 444,601
No accurate support bound | 331.226s | 60789s | 2,470,538 | 444,601
No equivalent points bound | 491.950s | 92.995s | 3,397,526 | 627,364
No one-step look ahead bound | 341.709s | 63.350s | 2,552,514 | 462,409

##### the old version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 312.042s | 57.737s | 3,347,339 | 598,330
No accurate support bound | 372.543s | 62.450 | 3,623,931 | 609,521
No equivalent points bound | 324.232s | 60.989s | 3,440,997 | 619,986
No one-step look ahead bound | 313.377s | 57.776s | 3,341,311 | 595,380
