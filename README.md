# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3), features ordered by gini reduction

#### lambda=0.001, MAXDEPTH=4, prior_metric="objective"

##### this version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 538.474s | 2.376s | 3,876,050 | 1,176
No accurate support bound | 540.971s | 2.281s | 3,876,050 | 1,176
No equivalent points bound | 976.324 | 2.371s | 6,323,384 | 1,372
No one-step look ahead bound | 569.242s | 2.347s | 4,055,553 | 1,176



#### lambda=0.0035, MAXDEPTH=4, prior_metric="objective"

##### this version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 327.208s | 60.741s | 2,470,538 | 444,601
No accurate support bound | 331.226s | 60789s | 2,470,538 | 444,601
No equivalent points bound | 491.950s | 92.995s | 3,397,526 | 627,364
No one-step look ahead bound | 341.709s | 63.350s | 2,552,514 | 462,409
