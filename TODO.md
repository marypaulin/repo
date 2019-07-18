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