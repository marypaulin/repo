# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### lambda=0.001, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | time | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------
All bounds | 1709s | 17,829,114 | 1,928,059
No support bound | 4649s | 40,360,485 | 5,440,663
No accurate support bound | 2096s | 20,603,649 | 1,928,059
No equivalent points bound | 1795s | 18,519,745 | 1,993,743
No one-step look ahead bound | 1703s | 17,813,838 | 1,928,223
