#undef NDEBUG

#include "coomatrix.hpp"
#include "bsr/blockmatrices.hpp"
#include "bsr/reorderingscaling.hpp"

using namespace blasted;

double computePerformanceMetric(const int job, const BSRMatrix<double,int,1>& mat)
{
	double perfmetric = 0;
	switch(job) {
	case 1:
		// check if the matrix has a zero diagonal
		{
			const size_t numzerodiags = mat.getNumZeroDiagonals();
			perfmetric = mat.dim() - numzerodiags;
		}
		break;
	case 2:
		// fallthrough
	case 3:
		perfmetric = mat.getAbsMinDiagonalEntry();
		break;
	case 4:
		perfmetric = mat.getDiagonalAbsSum();
		break;
	case 5:
		perfmetric = mat.getDiagonalProduct();
		break;
	default:
		throw std::runtime_error("Invalid job for MC64!");
	}
	return perfmetric;
}

int main(int argc, char *argv[])
{
	if(argc < 2) {
		std::cout << "Need mtx file name and MC64 job to perform (integer between 1 and 5).\n";
		std::exit(-1);
	}

	const std::string matfile = argv[1];
	const int job = std::stoi(argv[2]);
	assert(job >= 1);
	assert(job <= 5);

	BSRMatrix<double,int,1> omat = constructBSRMatrixFromMatrixMarketFile<double,int,1>(matfile);
	std::cout << "Matrix size = " << omat.dim() << std::endl;

	MC64 mc64(job);
	omat.computeOrderingScaling(mc64);

	BSRMatrix<double,int,1> mat1(omat);

	mat1.reorderScale(mc64, FORWARD);

	const double orig_perf = computePerformanceMetric(job, omat);
	const double reord_perf = computePerformanceMetric(job, mat1);
	std::cout << "Performance. Orig: " << orig_perf << ", reordered: " << reord_perf
	          << std::endl;
	assert(std::abs(reord_perf) >= std::abs(orig_perf));

	mat1.reorderScale(mc64, INVERSE);

	const double tol = (job == 5) ? 9e-9 : 1e-15;
	const std::array<bool,5> reseq = omat.isEqual(mat1, tol);

	assert(reseq[0]);
	assert(reseq[1]);
	assert(reseq[2]);
	assert(reseq[3]);
	assert(reseq[4]);

	return 0;
}
