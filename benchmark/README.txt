The benchmarking folder contains efforts to benchmark Theano against various
other systems.  Each subfolder corresponds to a particular type of computation,
and each sub-subfolder corresponds to the implementation of that computation in
with a particular software package.

Since there is a variety of benchmark problems and of software systems, there
isn't a standard for how to run the benchmark suite.
There is however a standard for how each benchmark should produce results.
Every benchmark run should produce one or more files with the results of
benchmarking. These files must end with extension '.bmark'.  These files must
have at least three lines each:

1) line 1 - description of computation/problem
2) line 2 - description of implementation/platform
3) line 3 - time required (in seconds)
4) line 4 - [optional] an estimated number of FLOPS performed (not necessarily same for all implementations of problem)

