/** \file testbsrmatrix.cpp
 * \brief Implementation of tests for block matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <float.h>
#include <memory>
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
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);

	const std::vector<double> x = readDenseMatrixMarket<double>(xvec);
	const std::vector<double> ans = readDenseMatrixMarket<double>(prodvec);

	std::unique_ptr<AbstractLinearOperator<double,int>> testmat;
	if(type == "view") {
		if(storageorder == "rowmajor")
			testmat = std::make_unique<BSRMatrixView<double,int,bs,RowMajor>>
				(move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(coom, storageorder)));
		else
			testmat = std::make_unique<BSRMatrixView<double,int,bs,ColMajor>>
				(move_to_const<double,int>(getSRMatrixFromCOO<double,int,bs>(coom, storageorder)));
	}
	else
		testmat = std::make_unique<BSRMatrix<double,int,bs>>
			(getSRMatrixFromCOO<double,int,bs>(coom, "rowmajor"));
	
	std::vector<double> y(testmat->dim());

	testmat->apply(x.data(), y.data());

	for(int i = 0; i < testmat->dim(); i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	return 0;
}

template int testBSRMatMult<3>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
template int testBSRMatMult<7>(const std::string type, const std::string storageorder,
		const std::string matfile, const std::string xvec, const std::string prodvec);
