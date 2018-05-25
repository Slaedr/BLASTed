#! /bin/bash

# Create a build directory and run this file from there.
# Needs BLASTED_DIR to be set to run.
# Needs PETSC_DIR and PETSC_ARCH to be set.
# Assumes the compilers to use are mpicc and mpicxx.

cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DWITH_PETSC=1 \
	  -DAVX=1 -DCMAKE_BUILD_TYPE=Release \
	  ${BLASTED_DIR}
