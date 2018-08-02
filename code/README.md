## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else
#### all 6907 data from compas-binary.csv, manually selected 5 features

##### time; number of trees in the queue (lambda=0.01)

number of feature | FIFO queue | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
vector |  | 4min44s; 1987208 | 3min40s; 1878542 | 2min37s; 961365 | 2min24s; 961709
mpz  |  | 3min38s; 1987208 | 2min48s; 1878542 | 1min43s; 961365 | 1min34s; 961709            
