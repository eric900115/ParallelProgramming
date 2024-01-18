#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

int cur_height = -1;
int* image;
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int ncpus;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void cal_row(int j) {

    double y0 = j * ((upper - lower) / height) + lower;
    int i = 0;
    double x0[2];
    int done[2];
    double length_squared[2] = {0};

    __m128d vec_y0 = _mm_loaddup_pd(&y0);
    __m128d vec_two = _mm_set1_pd(2.0);

    // use sse intrinsic to process 2 elements at the same time
    // Note that the register len of vector machine is 128 bits
    for (i = 0; i < (width-1); i += 2) {
        
        x0[0] = i * ((right - left) / width) + left;
        x0[1] = (i+1) * ((right - left) / width) + left;

        __m128d vec_x0 = _mm_loadu_pd(x0);

        done[0] = done[1] = 0;
        length_squared[0] = length_squared[1] = 0;

        __m128d vec_length_squared = _mm_setzero_pd();

        int repeats[2] = {0};
        int repeats_tmp[2] = {0};

        __m128d vec_x = _mm_setzero_pd();
        __m128d vec_y = _mm_setzero_pd();
        __m128d x_mul_x = _mm_mul_pd(vec_x, vec_x);
        __m128d y_mul_y = _mm_mul_pd(vec_y, vec_y);
        __m128d x_mul_y;


        while (!done[0] || !done[1]) {
            // temp = x * x - y * y + x0;
            __m128d vec_temp = _mm_add_pd( 
                _mm_sub_pd(x_mul_x, y_mul_y)
                , vec_x0);

            //y = 2 * x * y + y0;
            x_mul_y = _mm_mul_pd(vec_x, vec_y);
            vec_y = _mm_add_pd(_mm_add_pd(x_mul_y, x_mul_y), vec_y0);

            //x = temp;
            vec_x = vec_temp;

            // x*x and y*y
            x_mul_x = _mm_mul_pd(vec_x, vec_x);
            y_mul_y = _mm_mul_pd(vec_y, vec_y);

            //length_squared = x * x + y * y;
            vec_length_squared = _mm_add_pd(x_mul_x, y_mul_y);

            //_mm_storeu_pd(length_squared, vec_length_squared);


            repeats_tmp[0]++;
            repeats_tmp[1]++;

            if(!done[0]) {
                repeats[0] = repeats_tmp[0];
            }

            if(!done[1]) {
                repeats[1] = repeats_tmp[1];
            }

            if(!(repeats_tmp[0] < iters && vec_length_squared[0] < 4)) {
                done[0] = 1;
            }

            if(!(repeats_tmp[1] < iters && vec_length_squared[1] < 4)) {
                done[1] = 1;
            }
        }

        image[j * width + i] = repeats[0];
        image[j * width + i + 1] = repeats[1];
    }

    // processing the remaining element
    if(i == (width-1)) {
        double x0 = i * ((right - left) / width) + left;

        int repeats = 0;
        double x = 0;
        double y = 0;
        double length_squared = 0;
        double x_mul_x =  x * x;
        double y_mul_y = y * y;

        while (repeats < iters && length_squared < 4) {
            double temp = x_mul_x - y_mul_y + x0;
            y = 2 * x * y + y0;
            x = temp;
            x_mul_x = x * x;
            y_mul_y = y * y;
            length_squared = x_mul_x + y_mul_y;
            ++repeats;
        }
        image[j * width + i] = repeats;
    }
}


int main(int argc, char** argv) {

    int rank, size;

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* init omp & mpi */
	ncpus = CPU_COUNT(&cpu_set);

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    int *ans_image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    memset(image, 0, width * height * sizeof(int));
    memset(ans_image, 0, width * height * sizeof(int));

    /* omp parallel loop to solve mandelbrot set */
    #pragma omp parallel for schedule(dynamic)
    for(int x = rank; x < height; x+=size) {
        cal_row(x);
    }

    MPI_Reduce(image, ans_image, height*width, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);

    /* draw and cleanup */
    if(rank == 0) {
        write_png(filename, iters, width, height, ans_image);
    }

    free(image);

    MPI_Finalize();

    return 0;
}
