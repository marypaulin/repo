# TODO
 - PyVisualization of Problem Trace
 - Graph entire dependency tree from local dictionary
 - Edges are:
   - directed
   - grey for unknown
   - solid for optimal
   - dashed for pruned
 - Nodes are:
   - bold for in-local-queue
   - colour proportional to uncertainty

   
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