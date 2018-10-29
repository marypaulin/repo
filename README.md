# CORELS DT

corels_dt:
all 6907 data from compas-binary.csv

##### lambda=0.0035

##### Try 6 features(sex:Female, age:18-20,age:21-22, juvenile-crimes:=0, priors:2-3, priors:>3), MAXDEPTH = 5

##### # order x's columns according to gini_reduction
Algorithm variant (for function generate_new_splitleaf) | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Use the leaf with the minimum loss to find a pair of leaves as d0 | 2257.721s | 15.745s | 9,844,529 | 341,986
use removed_leaves as d0, unchanged_leaves as dp | 117.383s | 20.328s | 49,173 | 35,314