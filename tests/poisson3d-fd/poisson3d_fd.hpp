/** \file poisson3d_fd.hpp
 * \brief PETSc-based finite difference routines for Poisson Dirichlet problem on a Cartesian grid
 * \author Aditya Kashi
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#include <petscmat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cartmesh.hpp"

/// Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
/** Note that the values are only set for interior points.
 * \param f is the rhs vector
 * \param uexact is the exact solution
 */
PetscErrorCode computeRHS(const CartMesh *const m, DM da, PetscMPIInt rank, Vec f, Vec uexact);

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
PetscErrorCode computeLHS(const CartMesh *const m, DM da, PetscMPIInt rank, Mat A);

/// Computes L2 norm of a mesh function v
/** Assumes piecewise constant values in a dual cell around each node.
 * Note that the actual norm will only be returned by process 0; 
 * the other processes return only local norms.
 */
PetscReal computeNorm(const MPI_Comm comm, const CartMesh *const m, Vec v, DM da);
