# TODO
 - Ensure census is binary
 - Rerun for 2 - 32 on census
 - Add architecture diagram for IPC + Pruning Sequence
 - PyVisualization of Problem Trace
 - Log analysis for deadlock
 - Try LEAP server
 - graph visualization
 - Try arranging priority on Leviathan
 - Try Socket Implementation
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

- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 1 Worker, 1 Server
- Runtime: 14 s

- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 2 Worker, 1 Server
- Runtime: 13 s

- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 4 Worker, 1 Server
- Runtime: 31 s



- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 1 Worker, 1 Server
- Runtime: 60+ s (14 seconds?)

- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 2 Worker, 1 Server
- Runtime: 60+ s

- Commit Hash: 5d1e00b0a7a70bbab6242fc01b2b6c4c01d9a0d6 
- Machine: Local
- Network Composition: Pipe + Queue
- Processes: 4 Worker, 1 Server
- Runtime: 60+ s