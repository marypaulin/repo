# CORELS DT

corels_dt:
all 6907 data from compas-binary.csv

##### lambda=0.0035

##### Try 6 features(sex:Female, age:18-20,age:21-22, juvenile-crimes:=0, priors:2-3, priors:>3), MAXDEPTH = 5


##### # Use the first pair of leaves as d0
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
order x's columns according to gini_reduction | 1123.931s | 15.665s | 3,162,248 | 132,285
manually order of x's columns to be [5, 1, 4, 2, 3, 0] (this is the order in the optimal tree)  | 1004.697s | 14.418s | 2,671,375 | 81,090
manually order of x's columns to be [5, 0, 4, 1, 2, 3] | 569.988s | 13.845s | 1,568,859 | 100,001


##### # order x's columns according to gini_reduction
Algorithm variant (for function generate_new_splitleaf) | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Use the first pair of leaves as d0 | 1123.931s | 15.665s | 3,162,248 | 132,285
Use the leaf with the maximum loss to find a pair of leaves as d0 | 1550.824s | 16.360s | 3,648,455 | 118,063
Use the leaf with the minimum loss to find a pair of leaves as d0 | 454.551s | 16.187s | 1,009,994 | 155,108


##### # use the leaf with the minimum loss to find a pair of leaves as d0, order x's columns according to gini_reduction
Algorithm variant | total time | time to find the optimal tree | total number of trees pushed into the queue | when is the optimal tree pushed into the queue
  ------------- | ------------- | ------------- | -------------  | -------------
Don't reorder the features for each leaf | 454.551s | 16.187s | 1,009,994 | 155,108
Reorder features according to gini reduction for every leaf | 634.584s | 2.778s | 962,812 | 594
Reorder features according to gini reduction (only for leaves with 2 or less feature) | 435.271s | 3.129s | 841,856 | 846