Overview
--------

Message Passing Interface (MPI) is an approach to parallelism based on explicit communication between processes.  Python bindings to it are available via mpi4py.

Testing code that uses MPI can be awkward.  This package provides a pure-python mock MPI communicator that can be used to do so, without requiring actual MPI.  It uses multiprocessing to
start processes and makes a mocked Communicator (comm) object that takes the place of 
mpi4py.MPI.COMM_WORLD.


Installation
------------

The code is pure python; you can install it with:

```
    pip install mock_mpi
```

Usage
-----

Define a function that holds the code that should work in an MPI session:

```
    def function_to_test(comm):
        ...
```

If this were a real MPI session, it should work as:

```
    import mpi4py
    comm = MPI.COMM_WORLD
    function_to_test(comm)
```

which you would run with `mpiexec -n nproc ...`

To test the code without MPI, import mock_mpiexec from this package and run:

```
    mock_mpiexec(nproc, function_to_test)
```

Extra Arguments
---------------

You can also supply `args=[...]` and `kwargs={...}` to `mock_mpiexec` and they will be passed
to `function_to_test`:

```
    mock_mpiexec(nproc, function_to_test, args=[1,2,3], kwargs={'a':'b'})
```

mimics:

```
    import mpi4py
    comm = MPI.COMM_WORLD
    function_to_test(comm, 1, 2, 3, a='b')

```

This works if `args` and `kwargs` can be pickled (true for most basic python and numpy types).

Caveats
-------

1. This runs on python 3.5+
2. Absolutely no attention was paid to making this efficient.  This code
   is designed to be used for unit testing, not production.
3. Only the IntraComm object is currently mocked, not the many other features
   like operators, futures, topologies, windows, spawning ... 
3. Many methods are currently missing.  Only those below are currently implemented.  Others will raise ``NotImplementedError``
 - Get_rank
 - Get_size
 - Barrier
 - send - unlike real MPI this is non-blocking
 - Send - also non-blocking
 - recv
 - Recv
 - barrier
 - bcast
 - Bcast
 - scatter
 - reduce (only with the default sum operation)
 - Reduce (only with the default sum operation)
 - allreduce
 - Allreduce
 - alltoall
 - gather

Contributions and License
-------------------------

We would greatly welcome contributions, especially to fill in missing features.

The code and any contributions are released under a BSD 2-clause license (see the LICENSE file).
