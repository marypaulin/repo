# CORELS DT

corels_dt:
all 6907 data from compas-binary.csv

##### lambda=0.0035

##### Try 6 features(sex:Female, age:18-20,age:21-22, juvenile-crimes:=0, priors:2-3, priors:>3), MAXDEPTH = 5
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
This Version, prior_metric="objective" | 257.973s | 10.576s | 657,173 | 51,646
Last Version, prior_metric="objective"  | 1734.942s | 1114.673s | 48,858,192 | 31,007,691

##### 5 features (sex:Female, age:18-20, age:21-22, priors:2-3, priors:>3), MAXDEPTH = 5, Without similar support bound

##### This Version (Oct 16th), Multiple copies for each tree, each time split one or more leaves
##### For each copy, if we don't split some leaves, then in its children trees these leaves will not be split either. In this way, we can avoid duplications of new trees.
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
prior_metric="objective" | 4.774s | 2.276s | 1,672 | 516
prior_metric="bound" | 7.236s | 7.119s | 22,211 | 22,194
prior_metric="curiosity" | 5.902s | 1.636s | 7,172 | 137
prior_metric="gini" | 6.191s | 5.374s | 9,138 | 6,439


##### Last Version (Oct 9th), Multiple copies for each tree, each time split only one leaf
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
prior_metric="objective" | 4.960s | 3.650s | 47,780 | 26,561
prior_metric="bound" | 31.165s | 3.397s | 773,889 | 3,862
prior_metric="curiosity" | 6.391s | 5.025s | 86,712 | 46,334
prior_metric="gini" | 7.997s | 0.769s | 112,619 | 44