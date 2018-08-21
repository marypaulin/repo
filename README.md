# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

#### lambda=0.001, MAXDEPTH=4, prior_metric="objective"

##### this version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 603.782s | 2.384s | 4,285,056 | 1,370
No accurate support bound | 594.748s | 2.415s | 4,285,056 | 1,370
No equivalent points bound | 906.064s | 2.456s | 5,967,395 | 1,370
No one-step look ahead bound | 620.985s | 2.403s | 4,408,101 | 1,370


##### the old version:
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 586.285s | 1.753s | 6,307,044 | 672
No accurate support bound | 691.745s | 1.711s | 6,435,730 | 672
No equivalent points bound | 612.924s | 2.307s | 6,340,464 | 1,354
No one-step look ahead bound | 584.871s | 1.741s | 6,301,576 | 672
