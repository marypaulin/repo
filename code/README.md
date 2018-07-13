## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split, add theorem20+lemma2, exchange if and else

number of rule | Priority queue （least curiosity）
  ------------- | ------------- 
3 | 441ms
4 | 15.5s
5 | 36min 54s

    
### serial-Copy3:
#### multiple copies of tree with each only one leaf marked to be split
number of rule | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | ------------- | -------------
3 | 1.2s | 1.18s | 1.18s
4 | 3min 26s | 7min 54s | 6min 23s
            
