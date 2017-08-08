BLASTed
=======

Basic Linear Algebra Subprograms Threaded: A collection of sparse matrix containers and manipulators, along with certain linear algebra operations for use in solving partial differential equations. The main focus is on providing all low-level kernels required to implement solvers for large sparse linear systems of algebraic equations, in an object-oriented framework.

Currently, the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library is used to implement some operations.

Building
--------
Please set the environment variable EIGEN_DIR to the top-level Eigen directory before using cmake. Eg., in Bash, assuming you are currently inside the BLASTed top-level directory,

    export EIGEN_DIR=/path/to/Eigen
    mkdir build && cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Debug -DOMP=1
    make
    
to build the debug multi-threaded version.

