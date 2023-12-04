#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int rank, size;
    unsigned long long int start_id, end_id;
    unsigned long long int r = atoll(argv[1]);
    unsigned long long int k = atoll(argv[2]);
    unsigned long long int sum = 0;
    unsigned long long int sum_tmp = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_id = (unsigned long long int)((((double)(rank)) / (size)) * r);
    end_id = (unsigned long long int)((((double)(rank + 1)) / (size)) * r) - 1;
        
    for(unsigned long long int i = start_id; i <= end_id; i++) {
        sum_tmp = (sum_tmp + (unsigned long long int)ceil(sqrtl(r*r - i*i))) % k;
    }

    sum_tmp = (4 * sum_tmp) % k;

    if(rank == 0) {
        sum += sum_tmp;
        for(long long int i = 1; i < size; i++) {
            MPI_Recv(&sum_tmp, 1, MPI_UNSIGNED_LONG_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum = (sum + sum_tmp) % k;
        }
        printf("%llu\n", sum);
    }
    else {
        MPI_Send(&sum_tmp, 1, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
