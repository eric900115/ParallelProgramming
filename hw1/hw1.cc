#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

void merge_and_getLow(float *arr1, float *arr2, float *merged_array, unsigned int size1, unsigned int size2);

void merge_and_getHigh(float *arr1, float *arr2, float *merged_array, unsigned int size1, unsigned int size2);

void get_partition(int rank, int total_data_num, int proc_num, unsigned int* start_id, \
                    unsigned int* end_id, unsigned int* data_num, unsigned int* right_proc_data_num, \
                    unsigned int* left_proc_data_num, int* used_proc_num);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, proc_num;
    int used_proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    unsigned int total_data_num = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    double start_time, end_time;

    MPI_File input_file, output_file;
    float *data;
    float *merged_data;
    float *buff_data;
    float _buff;

    unsigned int start_id, end_id;
    unsigned int data_num;
    unsigned int left_proc_data_num;
    unsigned int right_proc_data_num;
    short is_sorted = 0, is_all_sorted = 0;

    // data partition
    get_partition(rank, total_data_num, proc_num, &start_id, \
                    &end_id, &data_num, &right_proc_data_num, \
                    &left_proc_data_num, &used_proc_num);
    
    // Read File
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    if(data_num > 0) {
        data = (float*)malloc(sizeof(float) * data_num);
        merged_data = (float*)malloc(sizeof(float) * 2 * (data_num + 1));
        buff_data = (float*)malloc(sizeof(float) * (data_num + 1));
        MPI_File_read_at(input_file, sizeof(float) * start_id, data, data_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&input_file);

    // Local Machine Sort
    if(data_num > 1) {
        boost::sort::spreadsort::spreadsort(data, data + data_num);
    }

    // Odd Even Sort
    while(!is_all_sorted) {
        
        is_all_sorted = 0;
        is_sorted = 1;

        // Even Phase
        if(data_num > 0) {

            if(rank % 2 == 0 && rank != (used_proc_num - 1)) {
                // send and receive data to rank + 1
                MPI_Sendrecv(&data[data_num-1], 1, MPI_FLOAT, rank+1, 1, \
                            &_buff, 1, MPI_FLOAT, rank+1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[data_num-1] > _buff) {
                    MPI_Sendrecv(data, data_num, MPI_FLOAT, rank+1, 1, \
                            buff_data, right_proc_data_num, MPI_FLOAT, rank+1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_getLow(data, buff_data, merged_data, data_num, right_proc_data_num);
                    is_sorted = 0;
                }
            }

            if(rank % 2 == 1) {
                // send and receive data from rank - 1
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank-1, 1, \
                            &_buff, 1, MPI_FLOAT, rank-1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[0] < _buff) {
                    MPI_Sendrecv(data, data_num, MPI_FLOAT, rank-1, 1, \
                            buff_data, left_proc_data_num, MPI_FLOAT, rank-1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_getHigh(data, buff_data, merged_data, data_num, left_proc_data_num);
                    is_sorted = 0;
                }
            }

            // Odd Phase
            if(rank % 2 == 0 && rank != 0) {
                // send and recieve from rank - 1
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank-1, 1, \
                            &_buff, 1, MPI_FLOAT, rank-1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[0] < _buff) {
                    MPI_Sendrecv(data, data_num, MPI_FLOAT, rank-1, 1, \
                            buff_data, left_proc_data_num, MPI_FLOAT, rank-1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_getHigh(data, buff_data, merged_data, data_num, left_proc_data_num);
                    is_sorted = 0;
                }
            }

            if((rank != used_proc_num - 1) && (rank % 2 == 1)) {
                // send and receive data from rank + 1
                MPI_Sendrecv(&data[data_num-1], 1, MPI_FLOAT, rank+1, 1, \
                            &_buff, 1, MPI_FLOAT, rank+1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[data_num-1] > _buff) {
                    MPI_Sendrecv(data, data_num, MPI_FLOAT, rank+1, 1, \
                            buff_data, right_proc_data_num, MPI_FLOAT, rank+1, \
                            MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_getLow(data, buff_data, merged_data, data_num, right_proc_data_num);
                    is_sorted = 0;
                }
            }
        }

        MPI_Allreduce(&is_sorted, &is_all_sorted, 1, MPI_SHORT, MPI_LAND, MPI_COMM_WORLD);
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    
    if(data_num > 0) {
        MPI_File_write_at(output_file, sizeof(float) * start_id, data, data_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&output_file);

    if(data_num > 0) {
        free(data);
        free(buff_data);
        free(merged_data);
    }

    MPI_Finalize();

    return 0;
}

void merge_and_getLow(float *arr1, float *arr2, float *merged_array, unsigned int size1, unsigned int size2) {

    // merge two sorted array arr1 and arr2
    // store the lower part of sorted array of arr1 and arr2 into merged_array and arr1
    // num of merged_array == size1

    unsigned int n = 0, i = 0, j = 0;

    while(n < size1 && i < size1 && j < size2) {
        if(arr1[i] < arr2[j]) {
            merged_array[n] = arr1[i];
            i++;
        }
        else {
            merged_array[n] = arr2[j];
            j++;
        }
        n++;
    }

    while(n < size1 && i < size1) {
        merged_array[n] = arr1[i];
        i++;
        n++;
    }

    while(n < size2 && j < size2) {
        merged_array[n] = arr2[j];
        j++;
        n++;
    }

    n = 0;
    while(n < size1) {
        arr1[n] = merged_array[n];
        n++;
    }

    return;
}

void merge_and_getHigh(float *arr1, float *arr2, float *merged_array, unsigned int size1, unsigned int size2) {

    // merge two sorted array arr1 and arr2
    // store the higher part of sorted array of arr1 and arr2 into merged_array and arr1
    // num of merged_array == size1

    long int n = size1 - 1, i = size1 - 1, j = size2 - 1;

    while((n >= 0) && (i >= 0) && (j >= 0)) {
        if(arr1[i] < arr2[j]) {
            merged_array[n] = arr2[j];
            j--;
        }
        else {
            merged_array[n] = arr1[i];
            i--;
        }
        n--;
    }

    while(n >= 0 && i >= 0) {
        merged_array[n] = arr1[i];
        i--;
        n--;
    }

    while(n >= 0 && j >= 0) {
        merged_array[n] = arr2[j];
        j--;
        n--;
    }

    n = 0;
    while(n < size1) {
        arr1[n] = merged_array[n];
        n++;
    }

    return;
}

void get_partition(int rank, int total_data_num, int proc_num, unsigned int* start_id, \
                    unsigned int* end_id, unsigned int* data_num, unsigned int* right_proc_data_num, \
                    unsigned int* left_proc_data_num, int* used_proc_num) {
    
    // get the info that the partition of current processor needs
    // start_id, end_id : offset of location of the file
    // data_num : number of data that the current processor has to process
    // right_proc_data_num : number of data that the rank+1 processor has to process
    // left_proc_data_num : number of data that the rank-1 processor has to process

    if(proc_num == 1) {
        *start_id = 0;
        *end_id = total_data_num - 1;
        *data_num = total_data_num;
        *left_proc_data_num = 0;
        *right_proc_data_num = 0;
        *used_proc_num = 1;
    }
    else if(total_data_num == proc_num) {
        *start_id = (unsigned int)((((double)(rank)) / (proc_num)) * total_data_num);
        *end_id = (unsigned int)((((double)(rank + 1)) / (proc_num)) * total_data_num) - 1;
        *data_num = 1;
        *used_proc_num = proc_num;

        if(rank == 0) {
            *left_proc_data_num = 0;
        }
        else {
            *left_proc_data_num = 1;
        }

        if(rank == (proc_num - 1)){
            *right_proc_data_num = 0;
        }
        else {
            *right_proc_data_num = 1;
        }
    }
    else if(total_data_num > proc_num) {
        if(rank < (total_data_num % proc_num)) {
            *data_num = total_data_num / proc_num + 1;
            *start_id = (*data_num) * rank;
            *end_id = *start_id + *data_num - 1;

            if(rank == 0) {
                *left_proc_data_num = 0;
            }
            else {
                *left_proc_data_num = total_data_num / proc_num + 1;
            }

            if(rank == (proc_num - 1)){
                *right_proc_data_num = 0;
            }
            else if(rank == (total_data_num % proc_num - 1)){
                *right_proc_data_num = total_data_num / proc_num;
            }
            else {
                *right_proc_data_num = total_data_num / proc_num + 1;
            }
        }
        else {
            *data_num = total_data_num / proc_num;
            *start_id = (total_data_num / proc_num + 1) * (total_data_num % proc_num) + (total_data_num / proc_num) * (rank - total_data_num % proc_num);
            *end_id = *start_id + *data_num - 1;

            if(rank == 0) {
                *left_proc_data_num = 0;
            }
            else if(rank == (total_data_num % proc_num)) {
                *left_proc_data_num = total_data_num / proc_num + 1;
            }
            else {
                *left_proc_data_num = total_data_num / proc_num;
            }

            if(rank == (proc_num - 1)){
                *right_proc_data_num = 0;
            }
            else {
                *right_proc_data_num = total_data_num / proc_num;
            }
        }
        *used_proc_num = proc_num;
    }
    else {
        if(rank < total_data_num) {
            *start_id = rank;
            *end_id = rank;
            *data_num = 1;

            if(rank == (total_data_num - 1)) {
                *right_proc_data_num = 0;
            }
            else {
                *right_proc_data_num = 1;
            }

            if(rank == 0) {
                *left_proc_data_num = 0;
            }
            else {
                *left_proc_data_num = 1;
            }
        }
        else {
            // no need to care start_id and end_id
            // since no data should be saved in this processor
            *start_id = total_data_num - 1; 
            *end_id = total_data_num - 1;
            *data_num = 0;
            *left_proc_data_num =  0;
            *right_proc_data_num = 0;
        }

        *used_proc_num = total_data_num;
    }
}

