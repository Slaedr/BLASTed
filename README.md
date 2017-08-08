BLASTed
=======

Basic Linear Algebra Subprograms Threaded: A collection of sparse matrix containers and manipulators, along with certain linear algebra operations for use in solving partial differential equations. The main focus is on providing all low-level kernels required to implement solvers for large sparse linear systems of algebraic equations, in an object-oriented framework.

Currently, the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library is used to implement some operations.

Building
--------
The library is currently header-only, mostly to make template instantiations possible. Just include the appropriate header in your code.

