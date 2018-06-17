#! /bin/bash

# In the BLASTed root directory, create a build directory called `build`
# and run this file from there.
# Needs EIGEN3_ROOT, PETSC_DIR and PETSC_ARCH to be set.
# May also need BOOST_ROOT to be set, if Boost is installed in a non-standard location.
# Assumes the compilers to use are mpicc and mpicxx.

cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DWITH_PETSC=1 \
	  -DAVX=1 -DCMAKE_BUILD_TYPE=Release ..
