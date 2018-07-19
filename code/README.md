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

##### time, lambda=0.04

number of feature | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 527ms | 454ms | 476ms | 462ms | 464ms
4 | 8.97s | 15.5s | 8.62s | 9.88s | 9.89s
5 | 10min53s | 36min 54s | 11min9s | 21min26s | 21min40s

##### time, lambda=0.01

number of feature | FIFO queue | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 1.06s | 706ms | 990ms | 902ms | 899ms 
4 | 1min22s | 4min52s | 1min45s | 5min | 5min09s 
5 | too long | too long | too long | too long | too long

##### number of trees, lambda=0.01

number of feature| FIFO queue | Priority queue （least curiosity） | Priority queue （least lbound）| Priority queue （least entropy） | Priority queue （least Gini index）| No objective bound, lookahead bound, equiv points bound; FIFO Queue 
  ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
3 | 158| 122 | 144 | 150 | 149 | 218 
4 | 4193| 8562 | 4727| 7927 | 7944 | 10226 


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
            
