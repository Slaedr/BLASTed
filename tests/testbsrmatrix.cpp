/** \file testbsrmatrix.cpp
 * \brief Implementation of tests for block matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <float.h>
#include <string>
#include <iomanip>
#include "testbsrmatrix.hpp"
#include "../src/coomatrix.hpp"

using namespace blasted;

template <int bs>
int testBSRMatMult(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec)
{
	RawBSRMatrix<double,int> rm;
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);
	if(storageorder == "rowmajor")
		coom.convertToBSR<bs,RowMajor>(&rm);
	else
		coom.convertToBSR<bs,ColMajor>(&rm);

	const double *const x = readDenseMatrixMarket<double>(xvec);
	const double *const ans = readDenseMatrixMarket<double>(prodvec);
	double *const y = new double[rm.nbrows*bs];

	AbstractLinearOperator<double,int>* testmat = nullptr;
	if(type == "view") {
		if(storageorder == "rowmajor")
			testmat = new BSRMatrixView<double,int,bs,RowMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,1,1);
		else
			testmat = new BSRMatrixView<double,int,bs,ColMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind,1,1);
	}
	else
		testmat = new BSRMatrix<double,int,bs>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind,1,1);
	
	testmat->apply(1.0, x, y);

	for(int i = 0; i < rm.nbrows*bs; i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	delete testmat;

	delete [] rm.browptr;
	delete [] rm.bcolind;
	delete [] rm.vals;
	delete [] rm.diagind;
	delete [] x;
	delete [] y;
	delete [] ans;

	return 0;
}

template int testBSRMatMult<3>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
template int testBSRMatMult<7>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
