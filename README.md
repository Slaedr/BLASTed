BLASTed
=======

Basic Linear Algebra Subprograms Threaded: A collection of sparse matrix containers and manipulators, along with certain linear algebra operations. The main focus is on providing thread-parallel low-level kernels required to implement solvers for large sparse linear systems of algebraic equations, in an object-oriented framework. In case of a distributed-nemory parallel solver, the operations implemented here are meant to be used in the subdomains' local preconditioners.

Building
--------
The library is currently header-only, mostly to make template instantiations possible. However, the PETSc interface and tests need to be built. The following programs and libraries are required:
- [CMake](https://cmake.org/) version 3.0 or later
- [Eigen](http://eigen.tuxfamily.org) version 3.3.4 or later
- [Boost](http://www.boost.org/)
- [PETSc](http://www.mcs.anl.gov/petsc/) and MPI are required to build the PETSc interface. PETSc 3.8 is required to run the tests.

Assuming that you are in the top-level BLASTed directory, type

    mkdir build && cd build

and then, to build a release version with AVX vectorization and the PETSc interface,

	cmake .. -DCMAKE\_BUILD\_TYPE=Release -DAVX=1 -DWITH\_PETSC=1 -DCMAKE\_CXX\_COMPILER=mpicxx

To build without the PETSc interface, `-DWITH_PETSC` should be removed. See the beginning of the top-level CMakeLists.txt file for all the options. To build,

    make

and to run the tests,

	make test

A C++ compiler with C++ 14 support is required; the build is known to work with GCC 5.4, GCC 6.4 and Intel 2017 in a GNU/Linux environment. To build in other enviroments, tweaking the CMakeLists.txt file will be required.

To build the [Doxygen](http://www.stack.nl/~dimitri/doxygen/) documentation,

    cd path/to/BLASTed/doc
    doxygen blasted_doxygen.cfg

This will build HTML documentation in a subdirectory called html in the current directory.

Finally, from the build directory, one can issue

    make tags
   
to generate a tags file for [easier navigation of the source code in Vim](http://vim.wikia.com/wiki/Browsing_programs_with_tags).

Usage
-----
For C++ codes, the library can be used header-only, if necessary. But for the most part, one would want to use it as a plugin for PETSc. For an example of that, please see the finite difference Poisson example in `tests/poisson3d-fd`.

Known issues
------------
The PETSc interface currently supports only scalar matrix storage in CSR format, ie., BSR matrices with block size 1.

