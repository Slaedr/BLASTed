Using BLASTed via the PETSc interface
=====================================

There are two main PETSc options controlling the use of BLASTed. These options are used for any preconditioner for which `-pc_type` is set to `shell`, assuming your code is set up to do that (see the next section).

* `-blasted_pc_type` sets the preconditioner to use. Each option below is affected by whether or not the BAIJ matrix type is being used - if so, the block version of the preconditioner is used, otherwise the regular scalar version is used.
  - `jacobi` Jacobi preconditioner
  - `sgs` Symmetric Gauss-Seidel preconditioner
  - `ilu0` ILU(0) preconditioner
  - `sapilu0` ILU(0) preconditioner with asynchronous factorization but sequential (forward- or back-substitution) application

* `-blasted_async_sweeps` An integer array specifying the number of asynchronous iterations ("sweeps") to use each time the preconditioner is built and applied. Eg.: `-blasted_async_sweeps 4,3` means the preconditioner is built using 4 asynchronous iterations (sweeps) while it is applied using 3 asynchronous sweeps. If not specified, the default of 1 sweep is used.

Changes needed in your source code
----------------------------------

The BLASTed setup, apply and cleanup functions will have to be set up as explained in the PETSc manual, section 4.4.7 on shell preconditioners. Keep in mind that since BLASTed preconditioners are meant to be local, they should be set as sub-preconditioners (`-sub_pc_type`) to a global preconditioner such as subdomain-block Jacobi (`-pc_type bjacobi`) or additive Schwarz (`-pc_type asm`). This means that in your code, the shell preconditioner to be set up must be obtained from the sub KSPs of the global preconditioner. For an example, see the finite difference Poisson example in `tests/poisson3d-fd/poisson3d.cpp`. In addition to the functions for setup, application and destruction, the context has to be supplied to PCSHELL. For BLASTed, this is the `Blasted_data` type defined in `include/blasted_petsc.h`. An object of this type must be created by the user, its `first_setup_done` member must be set to `false` and the block size `bs` must be set.

All this can be automated somewhat using the `setup_localpreconditioner_blasted` function in `include/blasted_petsc.h`. Just create a `Blasted_data` object and pass it to the function along with the global `KSP` in which to set BLASTed as the subdomain solver. If you use this function, the block size will be taken from the preconditioning operator associated with the `KSP` argument. Thus, `KSPSetOperators` must be called and the block size must be set (either using `MatSetBlockSize` or `-mat_block_size`) beforehand.
