/** \file runcoomatrixtest.cpp
 * \brief Execution of unittests for coordinate matrix module
 */

#undef NDEBUG
#include "testcoomatrix.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 2) {
		std::cout << "! Please specify the test. Options:\n";
		std::cout << " read\n convertCSR\n convertBSR3\n";
		std::abort();
	}
	std::string teststr = argv[1];

	int err = 0;

	if(teststr == "read")
	{
		if(argc < 4) {
			std::cout << "! After 'read', please give \n"
			<< "the file name containing the matrix in mtx format and\n"
			<< "the file name containing the sorted arrays that describe the matrix.\n";
			std::abort();
		}

		TestCOOMatrix tmat;
		int ierr = tmat.readCoordinateMatrix(argv[2],argv[3]);
		err = err || ierr;
	}
	else if(teststr == "convertCSR")
	{
		if(argc < 4) {
			std::cout << "! After 'convertCSR', please give \n"
			<< "the file name containing the matrix in mtx format and\n"
			<< "the file name containing the sorted arrays that describe the matrix.\n";
			std::abort();
		}

		COOMatrix<double,int> tmat;
		tmat.readMatrixMarket(argv[2]);

		TestCSRMatrix<double> cmat(1,1);
		BSRMatrix<double,int,1> * bmat = &cmat;

		tmat.convertToCSR(bmat);

		int ierr = cmat.testStorage(argv[3]);
		err = err || ierr;
	}
	else if(teststr == "convertBSR3")
	{
		if(argc < 4) {
			std::cout << "! After 'convertBSR3', please give \n"
			<< "the file name containing the matrix (of block-size 3) in mtx format and\n"
			<< "the file name containing the sorted arrays that describe the matrix.\n";
			std::abort();
		}

		int ierr = testConvertCOOToBSR<3,Eigen::ColMajor>(argv[2],argv[3]);
		err = err || ierr;
	}
	else {
		std::cout << "! The requested test is not available.\n";
		std::abort();
	}

	return err;
}
