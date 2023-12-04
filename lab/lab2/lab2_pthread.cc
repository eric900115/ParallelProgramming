#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

unsigned long long r, k;
unsigned long long r_square;
unsigned long long pixels = 0;
unsigned long long ncpus;
unsigned long long finished_cpu_cnt = 0;
pthread_mutex_t mutex;

void* calculate(void* threadid) {
    
    int* tid = (int*)threadid;
	unsigned long long start_id;
    unsigned long long end_id;

    if(ncpus == 1) {
        start_id = 0;
        end_id = r;
    }
    else if(r == ncpus) {
        start_id = *tid;
        end_id = *tid + 1;
    }
    else if(r < ncpus) {
        if(*tid < r) {
            start_id = *tid;
            end_id = *tid + 1;
        }
        else {
            start_id = 0;
            end_id = 0;
        }
    }
    else {
        if(*tid < (r % ncpus)){
            unsigned long long int data_num = r / ncpus + 1;
            start_id = (*tid) * data_num;
            end_id = start_id + data_num;
        }
        else {
            unsigned long long int data_num = r / ncpus;
            start_id = (r / ncpus + 1) * (r % ncpus) + (r / ncpus) * (*tid - r % ncpus);
            end_id = start_id + r / ncpus;
        }
    }

	unsigned long long sum_tmp = 0;

	for(unsigned long long int i = start_id; i < end_id; i++) {
        sum_tmp = (sum_tmp + (unsigned long long int)ceil(sqrtl(r_square - i*i)));
    }

    sum_tmp %= k;
	
    pthread_mutex_lock(&mutex);
	pixels = (pixels + sum_tmp);
    finished_cpu_cnt++;
	pthread_mutex_unlock(&mutex);

    if(finished_cpu_cnt == ncpus) {
        pixels %= k;
        printf("%llu\n", (4 * pixels) % k);
    }

    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {

	r = atoll(argv[1]);
	k = atoll(argv[2]);

    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
    
    int num_threads = ncpus;
    pthread_t threads[num_threads];
    int rc;
    int ID[num_threads];
    int t;

    r_square = r * r;

    pthread_mutex_init(&mutex, NULL);

    for (t = 0; t < num_threads; t++) {
        ID[t] = t;
        rc = pthread_create(&threads[t], NULL, calculate, (void*)&ID[t]);
    }

    pthread_exit(NULL);
}
