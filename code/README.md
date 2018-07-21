## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else
#### all 6907 data from compas-binary.csv

##### time; number of trees in the queue (lambda=0.04)

number of feature | FIFO queue  | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 593ms; 72 | 447ms; 62 |  |  | 
4 | 9.49s; 1679 | 8.58s; 2549 |  |  | 
5 | 2min5s; 27688 | 2min7s; 28470 |  |  | 

##### time; number of trees in the queue (lambda=0.01)

number of feature | FIFO queue | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 981ms; 158 | 653ms; 122 | 923ms; 144 | 836ms; 150 | 830ms; 149 
4 | 38.2s; 9517 | 31.1s; 8721 | 36.3s; 9191 | 29.2s; 8537 | 29.1s; 8552 
5 | 45min39s; 791088 | NA | NA | NA | NA

##### time; number of trees in the queue (lambda=0.005)

number of feature | FIFO queue | Priority queue （min curiosity） | Priority queue （min lbound）| Priority queue （min entropy） | Priority queue （min Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 1.46s; 246 | 649ms; 135 |  |  | 
4 | 2min33s; 38266 | 1min39s; 30748 |  |  | 
5 | NA | NA | NA | NA | NA
            
