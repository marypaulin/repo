## Use priority queue

### serial-Copy2:
#### one copy of tree with multiple leaves marked to be split

number of rule | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | -------------
3 | 1min 9s | 909ms | 874ms
4 | 1min 9s | 1min 11s | 1min 30s

    
### serial-Copy3:
#### multiple copies of tree with each only one leaf marked to be split
number of rule | FIFO queue  | Priority queue （least curiosity） | Priority queue （least lbound）
  ------------- | ------------- | -------------
3 | 1.2s | 1.18s | 1.18s
4 | 3min 26s | 7min 54s | 6min 23s
            
