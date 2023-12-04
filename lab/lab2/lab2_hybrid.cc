#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	int rank, size;
	unsigned long long int start_id, end_id;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long sum_pixels = 0;
	int chunk = 50;

	MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

	start_id = (unsigned long long int)((((double)(rank)) / (size)) * r);
  end_id = (unsigned long long int)((((double)(rank + 1)) / (size)) * r) - 1;

	#pragma omp parallel for schedule(guided, chunk) reduction(+:pixels)
	for (unsigned long long x = start_id; x <= end_id; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}

	pixels %= k;

	MPI_Allreduce(&pixels, &sum_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

	if(rank == 1) {
		printf("%llu\n", (4 * sum_pixels) % k);
	}

	MPI_Finalize();

	return 0;
}
