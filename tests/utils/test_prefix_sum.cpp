
#include "utils/prefix_sum.hpp"

#include <vector>
#include <random>

static void seq_prefix_sum(int *const counts, const int size)
{
	int partial_sum{0};
	for(int i = 0; i < size; i++) {
		int temp = counts[i];
		counts[i] = partial_sum;
		partial_sum += temp;
	}
}

int main(int argc, char *argv[])
{
	const int N = 202;
	std::ranlux48 engine(43); // 43 is arbirtrary
	std::uniform_int_distribution<int> dist(0,N);
	std::vector<int> counts(N);
	for(int i = 0; i < N; i++)
		counts[i] = dist(engine);

	std::vector<int> refcounts(counts);

	seq_prefix_sum(refcounts.data(), N);
	blasted::prefix_sum(counts.data(), N);

	for(int i = 0; i < N; i++) {
		if(counts[i] != refcounts[i]) {
			printf(" Pos %d: Ref val = %d, test val = %d.\n", i, refcounts[i], counts[i]);
			return -1;
		}
	}

	return 0;
}
