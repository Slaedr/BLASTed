/** \file
 * \brief Tests for miscellaneous helper routines
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <string>
#include "../src/helper_algorithms.hpp"

/// Tests \ref sortBlockInnerDimension
int testSortBlockInnerDimension(std::string testfile, std::string solnfile)
{
	constexpr int bs = 2;

	std::vector<int> cind;
	std::vector<double> vals;
	std::ifstream tf;
	tf.open(testfile);
	if(!tf) throw std::runtime_error("File does not exist!");

	int tbs, N;
	tf >> N >> tbs;
	assert(tbs == bs);
	cind.resize(N); vals.resize(N*bs*bs);

	for(int i = 0; i < N; i++)
		tf >> cind[i];
	for(int i = 0; i < N*bs*bs; i++)
		tf >> vals[i];
	tf.close();

	std::vector<int> scind(N), svals(N*bs*bs);
	std::ifstream sf(solnfile);
	if(!sf) throw std::runtime_error("File does not exist!");
	int ns;
	sf >> ns >> tbs; // reuse of tbs
	assert(N==ns);
	assert(bs==tbs);
	for(int i = 0; i < N; i++)
		sf >> scind[i];
	for(int i = 0; i < N*bs*bs; i++)
		sf >> svals[i];
	sf.close();

	blasted::internal::sortBlockInnerDimension<double,int,bs>(N, &cind[0], &vals[0]);

	for(int i = 0; i < N; i++) {
		std::cout << cind[i] << " " << scind[i] << std::endl;
		assert(cind[i] == scind[i]);
		for(int k = 0; k < bs*bs; k++)
			assert(vals[i*bs*bs+k] == svals[i*bs*bs+k]);
	}

	return 0;
}

int main(int argc, char *argv[])
{
	assert(argc >= 3);
	return testSortBlockInnerDimension(argv[1],argv[2]);
}
