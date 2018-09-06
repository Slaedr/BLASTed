/** \file testbsrmatrix.cpp
 * \brief Implementation of tests for block matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <float.h>
#include <string>
#include <iomanip>
#include <vector>

#include <coomatrix.hpp>
#include "testbsrmatrix.hpp"

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

	const std::vector<double> x = readDenseMatrixMarket<double>(xvec);
	const std::vector<double> ans = readDenseMatrixMarket<double>(prodvec);
	std::vector<double> y(rm.nbrows*bs);

	AbstractLinearOperator<double,int>* testmat = nullptr;
	if(type == "view") {
		if(storageorder == "rowmajor")
			testmat = new BSRMatrixView<double,int,bs,RowMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind);
		else
			testmat = new BSRMatrixView<double,int,bs,ColMajor>(rm.nbrows,
					rm.browptr,rm.bcolind,rm.vals,rm.diagind);
	}
	else
		testmat = new BSRMatrix<double,int,bs>(rm.nbrows,
				rm.browptr,rm.bcolind,rm.vals,rm.diagind);
	
	testmat->apply(x.data(), y.data());

	for(int i = 0; i < rm.nbrows*bs; i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	delete testmat;
	destroyRawBSRMatrix<double,int>(rm);

	return 0;
}

template int testBSRMatMult<3>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
template int testBSRMatMult<7>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
