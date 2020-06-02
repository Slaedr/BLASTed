/** \file blasted_petsc_io.cpp
 * \brief Implementation of some I/O operations involving PETSc objects
 * \author Aditya Kashi
 */

#include <cstring>
#include <vector>
#include <memory>

#include <utils/blasted_petsc_io.h>
#include <coomatrix.hpp>
#include <blockmatrices.hpp>

extern "C" {

PetscErrorCode readVecFromFile(const char *const file, MPI_Comm comm, Mat a, Vec *const v)
{
	PetscErrorCode ierr = 0;

	blasted::device_vector<PetscReal> array = blasted::readDenseMatrixMarket<PetscReal>(file);
	const PetscInt size = static_cast<PetscInt>(array.size());

	if(a) {
		ierr = MatCreateVecs(a, NULL, v); CHKERRQ(ierr);
	}
	else {
		ierr = VecCreate(comm, v); CHKERRQ(ierr);
		ierr = VecSetSizes(*v, PETSC_DECIDE, size); CHKERRQ(ierr);
		ierr = VecSetUp(*v); CHKERRQ(ierr);
	}

	std::vector<PetscInt> indices(size);
	for(PetscInt i = 0; i < size; i++)
		indices[i] = i;

	ierr = VecSetValues(*v, size, indices.data(), array.data(), INSERT_VALUES);
	CHKERRQ(ierr);

	ierr = VecAssemblyBegin(*v); CHKERRQ(ierr);
	ierr = VecAssemblyEnd(*v); CHKERRQ(ierr);

	return ierr;
}

/// Slow. readMatFromCOOFile_viaSR is MUCH faster for blocksize 1, and slightly faster otherwise
PetscErrorCode readMatFromCOOFile(const char *const file, MPI_Comm comm, Mat *const A)
{
	PetscErrorCode ierr = 0;
	PetscMPIInt rank, mpisize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &mpisize);
	if(rank == 0)
		printf("  Num procs = %d\n", mpisize);

	blasted::COOMatrix<PetscReal,PetscInt> cmat;
	cmat.readMatrixMarket(file);
	if(rank == 0)
		printf("  Dim of matrix is %d x %d\n", cmat.numrows(), cmat.numcols());

	ierr = MatCreate(comm, A); CHKERRQ(ierr);
	// set matrix type (and block size) from options
	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);
	PetscInt bs;
	ierr = MatGetBlockSize(*A,&bs); CHKERRQ(ierr);

	ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, cmat.numrows(), cmat.numcols()); CHKERRQ(ierr);

	// for default pre-allocation
	ierr = MatSetUp(*A); CHKERRQ(ierr);
	ierr = MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE); CHKERRQ(ierr); // we insert entries by row

	const std::vector<PetscInt>& rowptrs = cmat.getRowPtrs();
	const std::vector<blasted::Entry<PetscReal,PetscInt>>& entries = cmat.getEntries();

	std::cout << "  Rows:" << std::endl;
	//#pragma omp parallel for default(shared)
	for(PetscInt irow = 0; irow < cmat.numrows(); irow++)
	{
		if(irow != entries[rowptrs[irow]].rowind)
			throw std::runtime_error("Encountered invalid COO row indices!");
		const int ncols = rowptrs[irow+1]-rowptrs[irow];
		// if(irow % 100 == 0) {
		// 	std::cout << "  " << irow << ", nzcols = " << ncols << std::flush;
		// }
		std::vector<PetscInt> colinds(ncols);
		std::vector<PetscReal> values(ncols);

		//#pragma omp simd
		for(PetscInt jr = rowptrs[irow]; jr < rowptrs[irow+1]; jr++) {
			const int jlr = jr-rowptrs[irow];
			colinds[jlr] = entries[jr].colind;
			values[jlr] = entries[jr].value;
		}
		//#pragma omp critical
		{
			ierr = MatSetValues(*A, 1, &irow, ncols, &colinds[0], &values[0], INSERT_VALUES);
		}
	}

	CHKERRQ(ierr);

	std::cout << "  Beginning assembly of matrix..\n" << std::endl;
	ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	std::cout << "  Completed assembly of matrix.\n" << std::endl;

	return ierr;
}

PetscErrorCode readMatFromCOOFile_viaSR(const char *const file, MPI_Comm comm, Mat *const A)
{
	PetscErrorCode ierr = 0;
	PetscMPIInt rank, mpisize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &mpisize);
	if(rank == 0)
		printf(" Num procs = %d\n", mpisize);
	if(mpisize > 1) {
		throw std::runtime_error(" Currently only 1 rank is supported");
	}

	blasted::COOMatrix<PetscReal,PetscInt> cmat;
	cmat.readMatrixMarket(file);
	if(rank == 0)
		printf(" Dim of matrix is %d x %d\n", cmat.numrows(), cmat.numcols());

	ierr = MatCreate(comm, A); CHKERRQ(ierr);
	// set matrix type (and block size) from options
	ierr = MatSetFromOptions(*A); CHKERRQ(ierr);
	PetscInt bs;
	ierr = MatGetBlockSize(*A,&bs); CHKERRQ(ierr);

	std::unique_ptr<blasted::SRMatrixStorage<PetscReal,PetscInt>> rmat;
	switch(bs) {
		case 3:
			//cmat.convertToBSR<3,Eigen::ColMajor>(&rmat);
			rmat = std::make_unique<blasted::SRMatrixStorage<PetscReal,PetscInt>>
				(std::move(blasted::getSRMatrixFromCOO<PetscReal,PetscInt,3>(cmat,"colmajor")));
			break;
		case 4:
			//cmat.convertToBSR<4,Eigen::ColMajor>(&rmat);
			rmat = std::make_unique<blasted::SRMatrixStorage<PetscReal,PetscInt>>
				(std::move(blasted::getSRMatrixFromCOO<PetscReal,PetscInt,4>(cmat,"colmajor")));
			break;
		case 7:
			//cmat.convertToBSR<7,Eigen::ColMajor>(&rmat);
			rmat = std::make_unique<blasted::SRMatrixStorage<PetscReal,PetscInt>>
				(std::move(blasted::getSRMatrixFromCOO<PetscReal,PetscInt,7>(cmat,"colmajor")));
			break;
		default:
			printf("  readMatFromCOOFile: Block size %d is not supported; falling back to CSR\n", bs);
			//cmat.convertToCSR(&rmat);
			rmat = std::make_unique<blasted::SRMatrixStorage<PetscReal,PetscInt>>
				(std::move(blasted::getSRMatrixFromCOO<PetscReal,PetscInt,1>(cmat,"colmajor")));
			bs = 1;
			ierr = MatSetBlockSize(*A, bs); CHKERRQ(ierr);
	}

	std::cout << "  Converted matrix to (B)CSR." << std::endl;

	if(bs == 1) {
		ierr = MatCreateMPIAIJWithArrays(PETSC_COMM_SELF, rmat->nbrows, rmat->nbrows,rmat->nbrows,
		                                 rmat->nbrows, &rmat->browptr[0], &rmat->bcolind[0], &rmat->vals[0], A);
		CHKERRQ(ierr);
	}
	else {
		ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, cmat.numrows(), cmat.numcols()); CHKERRQ(ierr);

		// for default pre-allocation
		ierr = MatSetUp(*A); CHKERRQ(ierr);
		ierr = MatSetOption(*A, MAT_ROW_ORIENTED, PETSC_FALSE); CHKERRQ(ierr);
		for(PetscInt i = 0; i < rmat->nbrows; i++) {
			const PetscInt *const colinds = &rmat->bcolind[rmat->browptr[i]];
			const PetscReal *const values = &rmat->vals[rmat->browptr[i]*bs*bs];
			const PetscInt ncols = rmat->browptr[i+1]-rmat->browptr[i];
			if(bs == 1)
				ierr = MatSetValues(*A, 1, &i, ncols, colinds, values, INSERT_VALUES);
			else
				ierr = MatSetValuesBlocked(*A, 1, &i, ncols, colinds, values, INSERT_VALUES);
			CHKERRQ(ierr);
		}
	}

	std::cout << "  Beginning assembly of matrix..\n" << std::endl;
	ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	std::cout << "  Completed assembly of matrix.\n" << std::endl;

	return ierr;
}

}
