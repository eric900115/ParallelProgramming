#include <cstdio>
#include <cstdlib>
#include <pthread.h>

int V, E;
int ncpus;

int** D;

const int INF = (1 << 30) - 1;

pthread_barrier_t barrier;

void input(char* input_file) {

  FILE* file = fopen(input_file, "rb");

  int edge_data[3];

  fread(&V, sizeof(int), 1, file);
  fread(&E, sizeof(int), 1, file);

  D = (int**)malloc(V * sizeof(int*));

  for(int i = 0; i < V; i++) {
    D[i] = (int*)malloc(V * sizeof(int*));
  }

  for(int i = 0; i < V; i++) {
    for(int j = 0; j < V; j++) {
      if(i == j) {
        D[i][j] = 0;
      }
      else {
        D[i][j] = INF;
      }
    }
  }

  for(int i = 0; i < E; i++) {
    fread(edge_data, sizeof(int), 3, file);
    D[edge_data[0]][edge_data[1]] = edge_data[2];
  }

  fclose(file);

  return;
}

void output(char* output_file) {

  FILE* file = fopen(output_file, "w");

  for(int i = 0; i < V; i++) {
    fwrite(D[i], sizeof(int), V, file);
  }

  fclose(file);

  return;
}


void* FloydWarshall(void* args) {

  int* tid = (int*)args;

  for(int k = 0; k < V; k++) {

    for(int i = *tid; i < V; i += ncpus) {
      for(int j = 0; j < V; j++) {
        if(D[i][j] > (D[i][k] + D[k][j])) {
          D[i][j] = D[i][k] + D[k][j];
        }
      }
    }

    pthread_barrier_wait(&barrier);
  }

  pthread_exit(NULL);
}

int main(int argc, char* argv[]) {

  /* detect how many CPUs are available */
  cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

  int num_threads = ncpus;
  pthread_t threads[num_threads];
  int rc;
  int ID[num_threads];
  int t;

  pthread_barrier_init(&barrier, NULL, ncpus);

  /* input and preprocess data */
  input(argv[1]);

  /* create threads to solve Floyd Warshall */
  for (t = 0; t < num_threads; t++) {
    ID[t] = t;
    rc = pthread_create(&threads[t], NULL, FloydWarshall, (void*)&ID[t]);
  }

  /* join pthread */
  for (int i = 0; i < num_threads; ++i) {
	  pthread_join(threads[i], NULL);
  }

  /* output the result */
  output(argv[2]);

  pthread_exit(NULL);

}
