BLASTed
=======

Basic Linear Algebra Subprograms Threaded: A collection of sparse matrix containers and manipulators, along with certain linear algebra operations for use in solving partial differential equations. The main focus is on providing low-level kernels required to implement solvers for large sparse linear systems of algebraic equations, in an object-oriented framework.

Building
--------
The library is currently header-only, mostly to make template instantiations possible. Just include the appropriate header in your code. However, the PETSc interface and tests need to be built. The following programs and libraries are required:
- [Eigen](http://eigen.tuxfamily.org) version 3.3.4 or later
- [Boost](http://www.boost.org/)
- [CMake](https://cmake.org/) version 3.0 or later
Optionally, [PETSc](http://www.mcs.anl.gov/petsc/) version 3.8 or later is required to build the PETSc interface.

Assuming that you are in the top-level BLASTed directory, type

    mkdir build && cd build

and then, to build a release version with AVX vectorization and the PETSc interface,

	cmake .. -DCMAKE_BUILD_TYPE=Release -DAVX=1 -DWITH_PETSC=1

To build without the PETSc interface, `-DWITH_PETSC` should be removed. See the beginning of the top-level CMakeLists.txt file for all the options. To build,

    make

and to run the tests,

	cd tests
	ctest

A C++ compiler with C++ 11 support is required; the build is known to work with GCC 5.4, GCC 6.4 and Intel 2017 in a GNU/Linux environment. To build in other enviroments, tweaking the CMakeLists.txt file will be required.

To build the [Doxygen](http://www.stack.nl/~dimitri/doxygen/) documentation,

    cd path/to/BLASTed/doc
    doxygen blasted_doxygen.cfg

This will build HTML documentation in the current directory.

