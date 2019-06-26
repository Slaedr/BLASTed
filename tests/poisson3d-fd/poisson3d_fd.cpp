/** \file poisson3d_fd.cpp
 * \brief PETSc-based finite difference routines for Poisson Dirichlet problem on a Cartesian grid
 * \author Aditya Kashi
 *
 * Note that only zero Dirichlet BCs are currently supported.
 */

#include "poisson3d_fd.hpp"

#if 0
/// Gives the index of a point in the point grid collapsed to 1D
static inline PetscInt getFlattenedIndex(const CartMesh *const m, 
		const PetscInt i, const PetscInt j, const PetscInt k)
{
	return i + m->gnpoind(0)*j + m->gnpoind(0)*m->gnpoind(1)*k;
}

/// Gives the index of a point in the point grid collapsed to 1D
/** Assumes boundary points don't exist.
 * Returns -1 when passed a boundary point.
 * Make sure there's at least one interior point, or Bad Things (TM) may happen.
 */
static inline PetscInt getFlattenedInteriorIndex(const CartMesh *const m, 
		const PetscInt i, const PetscInt j, const PetscInt k)
{
	PetscInt retval = i-1 + (m->gnpoind(0)-2)*(j-1) + (m->gnpoind(0)-2)*(m->gnpoind(1)-2)*(k-1);
	if(i == 0 || i == m->gnpoind(0)-1 || j == 0 || j == m->gnpoind(1)-1 
			|| k == 0 || k == m->gnpoind(2)-1) 
	{
		//std::printf("getFlattenedInteriorIndex(): i, j, or k index corresponds to boundary node.
		//Flattened index = %d, returning -1\n", retval);
		return -1;
	}
	return retval;
}
#endif

/// Set RHS = 12*pi^2*sin(2pi*x)sin(2pi*y)sin(2pi*z) for u_exact = sin(2pi*x)sin(2pi*y)sin(2pi*z)
/** Note that the values are only set for interior points.
 * \param f is the rhs vector
 * \param uexact is the exact solution
 */
PetscErrorCode computeRHS(const CartMesh *const m, DM da, PetscMPIInt rank, Vec f, Vec uexact)
{
	PetscErrorCode ierr = 0;	
	if(rank == 0)
		printf("ComputeRHS: Starting\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	// get local data that can be accessed by global indices
	PetscReal *** rhs, *** uex;
	ierr = DMDAVecGetArray(da, f, (void*)&rhs); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da, uexact, (void*)&uex); CHKERRQ(ierr);

	// iterate over interior nodes
	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				rhs[k][j][i] = 12.0*PI*PI * std::sin(2*PI*m->gcoords(0,i))
					* std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
				uex[k][j][i] = std::sin(2*PI*m->gcoords(0,i))
					* std::sin(2*PI*m->gcoords(1,j))*std::sin(2*PI*m->gcoords(2,k));
			}
	
	DMDAVecRestoreArray(da, f, (void*)&rhs);
	DMDAVecRestoreArray(da, uexact, (void*)&uex);
	if(rank == 0)
		printf("ComputeRHS: Done\n");

	return ierr;
}

/// Set stiffness matrix corresponding to interior points
/** Inserts entries rowwise into the matrix.
 */
PetscErrorCode computeLHS(const CartMesh *const m, DM da, PetscMPIInt rank, Mat A)
{
	PetscErrorCode ierr = 0;	
	if(rank == 0)	
		printf("ComputeLHS: Setting values of the LHS matrix...\n");

	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	ierr = DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	CHKERRQ(ierr);

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal values[NSTENCIL];
				MatStencil cindices[NSTENCIL];
				MatStencil rindices[1];
				PetscInt n = NSTENCIL;
				PetscInt mm = 1;

				rindices[0] = {k,j,i,0};

				cindices[0] = {k-1,j,i,0};
				cindices[1] = {k,j-1,i,0};
				cindices[2] = {k,j,i-1,0};
				cindices[3] = {k,j,i,0};
				cindices[4] = {k,j,i+1,0};
				cindices[5] = {k,j+1,i,0};
				cindices[6] = {k+1,j,i,0};

				PetscInt I = i+1, J = j+1, K = k+1;		// 1-offset indices for mesh coords access
				
				values[2] = -1.0/( (m->gcoords(0,I)-m->gcoords(0,I-1)) 
						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[1] = -1.0/( (m->gcoords(1,J)-m->gcoords(1,J-1)) 
						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[0] = -1.0/( (m->gcoords(2,K)-m->gcoords(2,K-1)) 
						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				values[3] =  2.0/(m->gcoords(0,I+1)-m->gcoords(0,I-1))*
				  (1.0/(m->gcoords(0,I+1)-m->gcoords(0,I))+1.0/(m->gcoords(0,I)-m->gcoords(0,I-1)));
				values[3] += 2.0/(m->gcoords(1,J+1)-m->gcoords(1,J-1))*
				  (1.0/(m->gcoords(1,J+1)-m->gcoords(1,J))+1.0/(m->gcoords(1,J)-m->gcoords(1,J-1)));
				values[3] += 2.0/(m->gcoords(2,K+1)-m->gcoords(2,K-1))*
				  (1.0/(m->gcoords(2,K+1)-m->gcoords(2,K))+1.0/(m->gcoords(2,K)-m->gcoords(2,K-1)));

				values[4] = -1.0/( (m->gcoords(0,I+1)-m->gcoords(0,I)) 
						* 0.5*(m->gcoords(0,I+1)-m->gcoords(0,I-1)) );
				values[5] = -1.0/( (m->gcoords(1,J+1)-m->gcoords(1,J)) 
						* 0.5*(m->gcoords(1,J+1)-m->gcoords(1,J-1)) );
				values[6] = -1.0/( (m->gcoords(2,K+1)-m->gcoords(2,K)) 
						* 0.5*(m->gcoords(2,K+1)-m->gcoords(2,K-1)) );

				MatSetValuesStencil(A, mm, rindices, n, cindices, values, INSERT_VALUES);
				//if(rank == 0)
				//	printf("\tProcessed index %d, diag value = %f\n", rindices[0], values[3]);
			}

	if(rank == 0)
		printf("ComputeLHS: Done.\n");
	
	return ierr;
}

/// Computes L2 norm of a mesh function v
/** Assumes piecewise constant values in a dual cell around each node.
 */
PetscReal computeNorm(const MPI_Comm comm, const CartMesh *const m, Vec v, DM da)
{
	// get the starting global indices and sizes (in each direction) of the local mesh partition
	PetscInt start[NDIM], lsize[NDIM];
	DMDAGetCorners(da, &start[0], &start[1], &start[2], &lsize[0], &lsize[1], &lsize[2]);
	
	// get local data that can be accessed by global indices
	PetscReal *** vv;
	DMDAVecGetArray(da, v, &vv);

	PetscReal norm = 0, global_norm = 0;

	for(PetscInt k = start[2]; k < start[2]+lsize[2]; k++)
		for(PetscInt j = start[1]; j < start[1]+lsize[1]; j++)
			for(PetscInt i = start[0]; i < start[0]+lsize[0]; i++)
			{
				PetscReal vol = 1.0/8.0*(m->gcoords(0,i+1)-m->gcoords(0,i-1))
					*(m->gcoords(1,j+1)-m->gcoords(1,j-1))*(m->gcoords(2,k+1)-m->gcoords(2,k-1));
				norm += vv[k][j][i]*vv[k][j][i]*vol;
			}

	DMDAVecRestoreArray(da, v, &vv);

	// get global norm
	MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

	return global_norm;
}
