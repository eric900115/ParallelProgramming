#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long r_square = r * r;
	int max_threads = omp_get_max_threads();
	//unsigned long long tmp_sum[max_threads];
	int chunk = 100;

	#pragma omp parallel for schedule(dynamic, chunk) reduction(+:pixels)
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}

	pixels %= k;
	
	printf("%llu\n", (4 * pixels) % k);
}
