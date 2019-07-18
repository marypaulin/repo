# TODO
 - Run Scalability Analysis, Convergence, Memory Profile, Accuracy Comparison
 - Figure out catching of brokenpipe error
 - Create Architectural Diagram

 - Sci-Kit Model Interface (Unit Tests)
 - Cache Eviction

 - Add Similarity Bound (Unforunately the precision gains aren't high enough to cause enough pruning)
 - Asynchronous Pruning (Suffix Match Proof)

# Testing
`python3 -m unittest discover -s tests`
`python3 -m unittest tests/test_osdt.py`

# Debug History

- 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 Machine, Pipe + Queue, 1 Core, 14 s