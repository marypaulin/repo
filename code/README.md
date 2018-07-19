## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else
#### all 6907 data from compas-binary.csv

#### FIFO queue,  lambda=0.04
Algorithm variant | 3 features | 4 features
  ------------- | ------------- | ------------- 
all bounds | 527ms | 8.97s 
No lookahead bound | 799ms | 25.4s 
No equivalent points bound | 1.02s | 1min12s

##### lambda=0.04

number of feature | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 527ms | 454ms | 476ms | 462ms | 464ms
4 | 8.97s | 15.5s | 8.62s | 9.88s | 9.89s
5 | 10min53s | 36min 54s | 11min9s | 21min26s | 21min40s

##### lambda=0.01

number of feature | FIFO queue | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 586ms | 639ms | 543ms | 678ms | 688ms
4 | 43s | 5min13s | 52.8s | 3min31s | 3min32s
5 | too long | too long | too long | too long | too long

##### lambda=0.0025

number of feature | FIFO queue | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 492ms | 1.05s | 468ms | 982ms | 985ms
4 | 1min34s | 1h50min25s | 1min33s | 1h9min23s | 1h7min39s
5 | too long | too long | too long | too long | too long


#### before add theorem20+lemma2, exchange if and else
number of feature | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | -------------
3 | 909ms | 909ms | 874ms
4 | 1min 9s | 1min 11s | 1min 30s

    
### serial-Copy3:
#### multiple copies of tree with each only one leaf marked to be split
number of feature | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | -------------
3 | 1.2s | 1.18s | 1.18s
4 | 3min 26s | 7min 54s | 6min 23s
            
