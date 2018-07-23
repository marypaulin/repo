## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else
#### all 6907 data from compas-binary.csv

##### time; number of trees in the queue (lambda=0.04)

number of feature | FIFO queue  | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 577ms; 72 | 419ms; 62 | 519ms; 69 | 449ms; 56 | 443ms; 56
4 | 22.2s; 3967 | 18.5s; 3654 | 21.9s; 3979 | 11.8s; 2352 | 11.6s; 2349
5 | 12min9s; 140952 | 11min21s; 137222 | 12min15s; 143242 | 5min34s; 68837 | 5min22s; 68048

##### time; number of trees in the queue (lambda=0.01)

number of feature | FIFO queue | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 945ms; 158 | 630ms; 122 | 878ms; 144 | 782ms; 150 | 772ms; 149
4 | 1min3s; 14375 | 43.8s; 12069 | 52s; 12292 | 34.5s; 10106 | 34.1s; 10088
5 |  | 1h20min38s; 1581030 |  |  | 

##### time; number of trees in the queue (lambda=0.005)

number of feature | FIFO queue | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 |  |  |  |  | 
4 |  |  |  |  | 
5 |  |  |  |  | 
            
