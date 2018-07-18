## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else
##### lambda=0.04, all 6907 data from compas-binary.csv

number of rule | FIFO queue  | Priority queue （but set all curiosity=1）| Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 492ms | 430ms | 441ms | 453ms
4 | 8.73s | 5.49s | 15.5s | 8.47s
5 | 10min47s | 8min58s | 36min 54s | 11min9s

##### lambda=0.01

number of rule | FIFO queue  | Priority queue （but set all curiosity=1）| Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 586ms | 610ms | 639ms | 543ms
4 | 43s | 2min11s | 5min13s | 52.8s
5 | too long | too long | too long | too long

##### lambda=0.0025

number of rule | FIFO queue  | Priority queue （but set all curiosity=1）| Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 492ms | 865ms | 1.05s | 468ms
4 | 1min34s | 50min5s | 1h50min25s | 1min33s
5 | too long | too long | too long | too long



#### before add theorem20+lemma2, exchange if and else
number of rule | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | -------------
3 | 909ms | 909ms | 874ms
4 | 1min 9s | 1min 11s | 1min 30s

    
### serial-Copy3:
#### multiple copies of tree with each only one leaf marked to be split
number of rule | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | -------------
3 | 1.2s | 1.18s | 1.18s
4 | 3min 26s | 7min 54s | 6min 23s
            
