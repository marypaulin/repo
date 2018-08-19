# CORELS DT

## Use priority queue

### corels_dt:
#### one copy of tree with multiple leaves marked to be split 
#### all 6907 data from compas-binary.csv, manually selected 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3)

##### lambda=0.003, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 280s | 129s | 3,320,719 | 1,560,183
No one-step look ahead bound | 286s | 130s | 3,322,177 | 1,560,185


##### lambda=0.0025, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 283s | 130s | 3,323,508 | 1,562,784
No support bound | 1064s | 396s | 10,361,775 | 4,352,077
No accurate support bound | 350s | 167s | 3,620,708 | 1,762,458
No equivalent points bound | 298s | 142s | 3,438,022 | 1,652,860
No one-step look ahead bound | 290s | 133s | 3,321,505 | 1,560,933

##### lambda=0.001, MAXDEPTH=4, prior_metric="gini"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 604s | 88s | 6,338,385 | 847,854
No support bound | 1218s | 196s | 13,054,196 | 2,405,959
No accurate support bound | 700s | 85s | 6,452,854 | 847,854
No equivalent points bound | 613s | 89s | 6,406,716 | 856,644
No one-step look ahead bound | 595s | 85s | 6,324,274 | 843,378


##### lambda=0.001, MAXDEPTH=4, prior_metric="objective"
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
All bounds | 488.2s | 1.5s | 6,307,044 | 672
No support bound | 1135.8s | 1.5s | 12,776,226 | 699
No accurate support bound | 582.8s | 1.5s | 6,435,730 | 672
No equivalent points bound | 514.7s | 2.1s | 6,340,464 | 1,354
No one-step look ahead bound | 482.8s | 1.5s | 6,301,576 | 672


##### lambda=0.001, MAXDEPTH=4, all bounds
metric of the priority queue| total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
gini | 584.2s | 84.3s | 6,338,385 | 847,854
entropy | 631.2s | 90.8s | 6,340,171 | 850,025
curiosity | 556.9s | 22.6s | 6,317,785 | 258,730
lower bound | 575.6s | 151.6s | 6,158,531 | 2,393,102
objective | 488.2s | 1.5s | 6,307,044 | 672
