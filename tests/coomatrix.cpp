/** \file coomatrix.coo
 * \brief Execution of unittests for coordinate matrix module
 */

#include "testcoomatrix.hpp"

int main(const int argc, const char *const argv[])
{
	if(argc < 2) {
		std::cout << "! Please specify the test. Options:\n";
		std::cout << " read\n convertCSR\n convertBSR\n";
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
		std::cout << "Not implemented yet..\n";
	}
	else if(teststr == "convertBSR")
	{
		std::cout << "Not implemented yet..\n";
	}
	else {
		std::cout << "! The requested test is not available.\n";
		std::abort();
	}

	return err;
}
