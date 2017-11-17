/** \file testbsrmatrix.hpp
 * \brief Tests for block matrix operations
 * \author Aditya Kashi
 */

#include "../src/blockmatrices.hpp"

using namespace blasted;

template <int bs>
class TestBSRMatrix : public BSRMatrix<double,int,bs>
{
public:
	TestBSRMatrix(const int nbuildsweeps, const int napplysweeps);
};
