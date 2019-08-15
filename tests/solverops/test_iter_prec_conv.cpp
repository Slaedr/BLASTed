/** \file
 * \brief Driver for testing the convergence of iterative preconditioners
 */

#undef NDEBUG

#ifdef _OPENMP
#include <omp.h>
#endif
#include "srmatrixdefs.hpp"
#include "ilu_pattern.hpp"
#include "device_container.hpp"
#include "../poisson3d-fd/poisson_setup.h"
#include "utils/cmdoptions.hpp"
#include "utils/mpiutils.hpp"
#include "../testutils.hpp"

using namespace blasted;

template <int bs>
int test_ilu_convergence(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                         const double tol, const int maxsweeps,
                         const int thread_chunk_size, const std::string initialization);

template <int bs>
int test_async_triangular_solve(const CRawBSRMatrix<double,int>& mat, const ILUPositions<int>& plist,
                                const double tol, const int maxiter, const int thread_chunk_size,
                                const std::string initialization);

int main(int argc, char *argv[])
{
	if(argc < 3) {
		printf(" ! Please provide the test type ('ailu','atriangular') and source ('poisson','file')\n");
		exit(-1);
	}

	PetscInitialize(&argc, &argv, NULL, NULL);
	const int rank = get_mpi_rank(PETSC_COMM_WORLD);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	const std::string testtype = argv[1];
	DiscreteLinearProblem dlp = generateDiscreteProblem(argc, argv, 2, false);

	int ierr = 0;
	const int bs = getBlockSize(dlp.lhs);
	printf(" Input matrix: block size = %d\n", bs);

	const int maxsweeps = parsePetscCmd_int("-max_sweeps");
	const double tol = parseOptionalPetscCmd_real("-tolerance", 1e-25);
	const int thread_chunk_size = parsePetscCmd_int("-blasted_thread_chunk_size");
	const std::string initialization = parsePetscCmd_string("-initialization", 20);

	const SRMatrixStorage<const double,const int> smat = wrapLocalPetscMat(dlp.lhs, bs);
	printf(" Input problem: Dimension = %d, nnz = %d\n", smat.nbrows*bs, smat.nnzb*bs*bs);
	assert(smat.nnzb == smat.browptr[smat.nbrows]);

	const CRawBSRMatrix<double,int> mat = createRawView(std::move(smat));
	assert(mat.nnzb == mat.browptr[mat.nbrows]);

	const ILUPositions<int> plist = compute_ILU_positions_CSR_CSR(&mat);

	if(testtype == "ailu")
		switch(bs) {
		case 1:
			ierr = test_ilu_convergence<1>(mat, plist, tol, maxsweeps, thread_chunk_size, initialization);
			break;
		case 4:
			ierr = test_ilu_convergence<4>(mat, plist, tol, maxsweeps, thread_chunk_size, initialization);
			break;
		default:
			throw std::out_of_range("Block size " + std::to_string(bs) + " not supported!");
		}
	else if(testtype == "triangular")
		switch(bs) {
		case 1:
			ierr = test_async_triangular_solve<1>(mat, plist, tol, maxsweeps, thread_chunk_size,
			                                      initialization);
			break;
		case 4:
			ierr = test_async_triangular_solve<4>(mat, plist, tol, maxsweeps, thread_chunk_size,
			                                      initialization);
			break;
		default:
			throw std::out_of_range("Block size " + std::to_string(bs) + " not supported!");
		}
	else
		throw std::runtime_error("Test type not supported!");

	ierr = destroyDiscreteLinearProblem(&dlp); CHKERRQ(ierr);
	ierr = PetscFinalize();
	return ierr;
}
