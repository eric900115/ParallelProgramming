#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

//======================
#define DEV_NO 0
cudaDeviceProp prop;

#define BlockingFactor 64
#define MAX_NUM_THREAD 1024 // Maximum nuber of threads per block
#define MEM_y_Offest 16 // 1024 / 64
#define Half_BF 32

// V : # of Vertex
// E : # of Edge
// n : dimension of Dist is n x n (n is used for padding the original graph) 
int n, V, E;
int* Dist;

const int INF = (1 << 30) - 1;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    if(V % 256 == 0) {
        n = V;
    }
    else {
        n = (V / 256 + 1) * 256;
    }

    Dist = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        }
    }

    if(n == V) {
        fwrite(Dist, sizeof(int), V * V, outfile);
    }
    else {
        for(int i = 0; i < V; i++) {
            fwrite(Dist + i * n, sizeof(int), V, outfile);
        }
    }

    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void blockedFW_Phase1(int *D, unsigned int round, unsigned int V) {

    int x = threadIdx.x;
    int y = threadIdx.y;

    // the address of required data in global memory
    int global_x = x + round * BlockingFactor;
    int global_y = y + round * BlockingFactor;

    // Cache the current data block being calculated
    __shared__ int shared_D[BlockingFactor][BlockingFactor];

    // load data from golbal memory to shared memory
    shared_D[y][x] = D[global_y * V + global_x];
    shared_D[y + Half_BF][x] = D[(global_y + Half_BF) * V + global_x];
    shared_D[y][x + Half_BF] = D[global_y * V + global_x + Half_BF];
    shared_D[y + Half_BF][x + Half_BF] = D[(global_y + Half_BF) * V + global_x + Half_BF];

    __syncthreads();

    // execute phase 1 of Blocked FW
    #pragma unroll 64
    for(int i = 0; i < BlockingFactor; i++) {
        shared_D[y][x] = min(shared_D[y][x],
            shared_D[y][i] + shared_D[i][x]);

        shared_D[y + Half_BF][x] = min(shared_D[y + Half_BF][x], 
            shared_D[y + Half_BF][i] + shared_D[i][x]);

        shared_D[y][x + Half_BF] = min(shared_D[y][x + Half_BF],
            shared_D[y][i] + shared_D[i][x + Half_BF]);

        shared_D[y + Half_BF][x + Half_BF] = min(shared_D[y + Half_BF][x + Half_BF],
            shared_D[y + Half_BF][i] + shared_D[i][x + Half_BF]);

        __syncthreads();
    }

    // store data from shared mem to global mem
    D[global_y * V + global_x] = shared_D[y][x];
    D[(global_y + Half_BF) * V + global_x] = shared_D[y + Half_BF][x];
    D[global_y * V + global_x + Half_BF] = shared_D[y][x + Half_BF];
    D[(global_y + Half_BF) * V + global_x + Half_BF] = shared_D[y + Half_BF][x + Half_BF];
}

__global__ void blockedFW_Phase2(int *D, unsigned int round, unsigned int V) {

    if(blockIdx.y == round)
        return;

    int x = threadIdx.x;
    int y = threadIdx.y;

    int global_x, global_y;
    int col_x, col_y, row_x, row_y;
    int blockID = blockIdx.y;

    __shared__ int shared_Pivot_Block[BlockingFactor][BlockingFactor];
    __shared__ int shared_Pivot_Col[BlockingFactor][BlockingFactor];
    __shared__ int shared_Pivot_Row[BlockingFactor][BlockingFactor];

    // calculate memory access offset
    global_x = x + round * BlockingFactor;
    global_y = y + round * BlockingFactor;
    col_x = x + round * BlockingFactor;
    col_y = y + blockID * BlockingFactor;
    row_x = x + blockID * BlockingFactor;
    row_y = y + round * BlockingFactor;

    // load data to shared memory
    shared_Pivot_Block[y][x] = D[global_y * V + global_x];
    shared_Pivot_Block[y][x + Half_BF] = D[global_y * V + global_x + Half_BF];
    shared_Pivot_Block[y + Half_BF][x] = D[(global_y + Half_BF) * V + global_x];
    shared_Pivot_Block[y + Half_BF][x + Half_BF] = D[(global_y + Half_BF) * V + global_x + Half_BF];

    shared_Pivot_Col[y][x] = D[col_y * V + col_x];
    shared_Pivot_Col[y][x + Half_BF] = D[col_y * V + col_x + Half_BF];
    shared_Pivot_Col[y + Half_BF][x] = D[(col_y + Half_BF) * V + col_x];
    shared_Pivot_Col[y + Half_BF][x + Half_BF] = D[(col_y + Half_BF) * V + col_x + Half_BF];

    shared_Pivot_Row[y][x] = D[row_y * V + row_x];
    shared_Pivot_Row[y][x + Half_BF] = D[row_y * V + row_x + Half_BF];
    shared_Pivot_Row[y + Half_BF][x] = D[(row_y + Half_BF) * V + row_x];
    shared_Pivot_Row[y + Half_BF][x + Half_BF] = D[(row_y + Half_BF) * V + row_x + Half_BF];

    __syncthreads();

    // Calculate Blocked Folyd Warshall
    #pragma unroll 64
    for(int k = 0; k < BlockingFactor; k++) {

        // Calculate Pivot Column
        shared_Pivot_Col[y][x] = min(shared_Pivot_Col[y][x], 
            shared_Pivot_Col[y][k] + shared_Pivot_Block[k][x]);

        shared_Pivot_Col[y][x + Half_BF] = min(shared_Pivot_Col[y][x + Half_BF], 
            shared_Pivot_Col[y][k] + shared_Pivot_Block[k][x + Half_BF]);

        shared_Pivot_Col[y + Half_BF][x] = min(shared_Pivot_Col[y + Half_BF][x], 
            shared_Pivot_Col[y + Half_BF][k] + shared_Pivot_Block[k][x]);

        shared_Pivot_Col[y + Half_BF][x + Half_BF] = min(shared_Pivot_Col[y + Half_BF][x + Half_BF], 
            shared_Pivot_Col[y + Half_BF][k] + shared_Pivot_Block[k][x + Half_BF]);

        // Calculate Pivot Row
        shared_Pivot_Row[y][x] = min(shared_Pivot_Row[y][x], 
            shared_Pivot_Block[y][k] + shared_Pivot_Row[k][x]);

        shared_Pivot_Row[y][x + Half_BF] = min(shared_Pivot_Row[y][x + Half_BF], 
            shared_Pivot_Block[y][k] + shared_Pivot_Row[k][x + Half_BF]);

        shared_Pivot_Row[y + Half_BF][x] = min(shared_Pivot_Row[y + Half_BF][x], 
            shared_Pivot_Block[y + Half_BF][k] + shared_Pivot_Row[k][x]);

        shared_Pivot_Row[y + Half_BF][x + Half_BF] = min(shared_Pivot_Row[y + Half_BF][x + Half_BF], 
            shared_Pivot_Block[y + Half_BF][k] + shared_Pivot_Row[k][x + Half_BF]);

    }

    // store data to global memory
    D[col_y * V + col_x] = shared_Pivot_Col[y][x];
    D[col_y * V + col_x + Half_BF] = shared_Pivot_Col[y][x + Half_BF];
    D[(col_y + Half_BF) * V + col_x] = shared_Pivot_Col[y + Half_BF][x];
    D[(col_y + Half_BF) * V + col_x + Half_BF] = shared_Pivot_Col[y + Half_BF][x + Half_BF];

    D[row_y * V + row_x] = shared_Pivot_Row[y][x];
    D[row_y * V + row_x + Half_BF] = shared_Pivot_Row[y][x + Half_BF];
    D[(row_y + Half_BF) * V + row_x] = shared_Pivot_Row[y + Half_BF][x];
    D[(row_y + Half_BF) * V + row_x + Half_BF] = shared_Pivot_Row[y + Half_BF][x + Half_BF];
}

__global__ void blockedFW_Phase3(int *D, unsigned int round, unsigned int V, unsigned int offset) {

    // process block(y, x)
    // block(y, x) depends on block(y, round) and block(round, x)

    if((blockIdx.x == round) || (blockIdx.y + offset == round))
        return;

    int data_0, data_1, data_2, data_3;
    int x, y;
    int global_x, global_y;
    int col_x, col_y, row_x, row_y;

    __shared__ int shared_Pivot_Col[BlockingFactor][BlockingFactor];
    __shared__ int shared_Pivot_Row[BlockingFactor][BlockingFactor];

    // calculate memory access offset
    x = threadIdx.x;
    y = threadIdx.y;
    global_x = x + blockIdx.x * BlockingFactor;
    global_y = y + (blockIdx.y + offset) * BlockingFactor;
    col_x = x + round * BlockingFactor;
    col_y = y + (blockIdx.y + + offset) * BlockingFactor;
    row_x = x + blockIdx.x * BlockingFactor;
    row_y = y + round * BlockingFactor;

    // load data to non global memory
    shared_Pivot_Col[y][x] = D[col_y * V + col_x];
    shared_Pivot_Col[y][x + Half_BF] = D[col_y * V + col_x + Half_BF];
    shared_Pivot_Col[y + Half_BF][x] = D[(col_y + Half_BF) * V + col_x];
    shared_Pivot_Col[y + Half_BF][x + Half_BF] = D[(col_y + Half_BF) * V + col_x + Half_BF];

    shared_Pivot_Row[y][x] = D[row_y * V + row_x];
    shared_Pivot_Row[y][x + Half_BF] = D[row_y * V + row_x + Half_BF];
    shared_Pivot_Row[y + Half_BF][x] = D[(row_y + Half_BF) * V + row_x];
    shared_Pivot_Row[y + Half_BF][x + Half_BF] = D[(row_y + Half_BF) * V + row_x + Half_BF];

    __syncthreads();

    data_0 = D[global_y * V + global_x];
    data_1 = D[(global_y + Half_BF) * V + global_x];
    data_2 = D[global_y * V + global_x + Half_BF];
    data_3 = D[(global_y + Half_BF) * V + global_x + Half_BF];

    // calculation of Blocked FW
    #pragma unroll 64
    for(int k = 0; k < BlockingFactor; k++) {
        data_0 = min(data_0, 
            shared_Pivot_Col[y][k] + shared_Pivot_Row[k][x]);
        data_1 = min(data_1, 
            shared_Pivot_Col[y + Half_BF][k] + shared_Pivot_Row[k][x]);
        data_2 = min(data_2, 
            shared_Pivot_Col[y][k] + shared_Pivot_Row[k][x + Half_BF]);
        data_3 = min(data_3, 
            shared_Pivot_Col[y + Half_BF][k] + shared_Pivot_Row[k][x + Half_BF]);
    }

    // store data to global memory
    D[global_y * V + global_x] = data_0;
    D[(global_y + Half_BF) * V + global_x] = data_1;
    D[global_y * V + global_x + Half_BF] = data_2;
    D[(global_y + Half_BF) * V + global_x + Half_BF] = data_3;
}

int main(int argc, char* argv[]) {

    input(argv[1]);

    int *device_dist[2];

    #pragma omp parallel num_threads(2)
    {
        unsigned int offset;

        unsigned int cpu_thread_id = omp_get_thread_num();

        cudaSetDevice(cpu_thread_id);

        cudaMalloc((void **) &device_dist[cpu_thread_id], n * n * sizeof(int));

        cudaMemcpy(device_dist[cpu_thread_id], Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

        // decide to use how many blocks and threads
        dim3 blockNum1(1, 1);
        dim3 blockNum2(1, n / BlockingFactor);
        dim3 blockNum3(n / BlockingFactor, n / BlockingFactor / 2);
        dim3 threadNum(32, 32);

        if(cpu_thread_id == 0) {
            offset = 0;
        }
        else {
            int num_blocks = n / BlockingFactor;
            offset = num_blocks / 2;
            if(num_blocks % 2 != 0) {
                blockNum3.y += 1;
            }
        }

        // execute blocked FW on device
        for(int round = 0; round < (n / BlockingFactor); round++) {
            blockedFW_Phase1 <<<blockNum1, threadNum>>> (device_dist[cpu_thread_id], round, n);
            blockedFW_Phase2 <<<blockNum2, threadNum>>> (device_dist[cpu_thread_id], round, n);
            blockedFW_Phase3 <<<blockNum3, threadNum>>> (device_dist[cpu_thread_id], round, n, offset);

            #pragma omp barrier

            if(((round + 1) >= offset) && ((round + 1) < (offset + blockNum3.y))) {
                if(cpu_thread_id == 0) {
                    int addr_offset = (round + 1) * BlockingFactor * n;
                    cudaMemcpyPeer(device_dist[1] + addr_offset, 1, device_dist[0] + addr_offset, 0, BlockingFactor * n * sizeof(int));
                }
                else {
                    int addr_offset = (round + 1) * BlockingFactor * n;
                    cudaMemcpyPeer(device_dist[0] + addr_offset, 0, device_dist[1] + addr_offset, 1, BlockingFactor * n * sizeof(int));
                }
            }

            #pragma omp barrier
        }

        if(cpu_thread_id == 0) {
            cudaMemcpy(Dist, device_dist[0], BlockingFactor * blockNum3.y * n * sizeof(int), cudaMemcpyDeviceToHost);
        }
        else {
            int addr_offset = offset * BlockingFactor * n;
            cudaMemcpy(Dist + addr_offset, device_dist[1] + addr_offset, BlockingFactor * blockNum3.y * n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        #pragma omp barrier

        cudaFree(device_dist[cpu_thread_id]);
    }

    // output data
    output(argv[2]);

    /*for(int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++)
            printf("%d ", Dist[i * n + j]);
        printf("\n");
    }*/

    return 0;
}