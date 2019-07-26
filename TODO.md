# TODO
 - Dependency chains spread across many processes are causing issues

 - Decide how to get locality for dependency chains (reduce synchronization)
 - Fix path pruning which is killing termination

 - Benchmark Julia IPC
 - Adaptive Compression

 - Sci-Kit Model Interface (Unit Tests)
 - Cache Eviction

 - Add Similarity Bound (Unforunately the precision gains aren't high enough to cause enough pruning)
 - Asynchronous Pruning (Suffix Match Proof)

# Testing
`python3 -m unittest discover -s tests`
`python3 -m unittest tests/test_osdt.py`

# Notes
 - It seems the 