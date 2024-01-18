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
#include <pthread.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

pthread_mutex_t mutex;

int cur_height = -1;
int* image;
int iters;
double left;
double right;
double lower;
double upper;
double y0_0;
double x0_0;
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

    double y0 = j * y0_0 + lower;
    int i = 1;
    int i0 = 0, i1 = 1;
    double x0[2];
    int done[2];

    __m128d vec_y0 = _mm_loaddup_pd(&y0);

    x0[0] = left;
    x0[1] = x0_0 + left;

    __m128d vec_x0 = _mm_loadu_pd(x0);

    done[0] = done[1] = 0;

    __m128d vec_length_squared = _mm_setzero_pd();

    int repeats[2] = {0};

    __m128d vec_x = _mm_setzero_pd();
    __m128d vec_y = _mm_setzero_pd();
    __m128d x_mul_x = _mm_mul_pd(vec_x, vec_x);
    __m128d y_mul_y = _mm_mul_pd(vec_y, vec_y);
    __m128d x_mul_y;

    // use sse intrinsic to process 2 elements at the same time
    // Note that the register len of vector machine is 128 bits
    while(i0 < width || i1 < width) {
        // temp = x * x - y * y + x0;
        __m128d vec_temp = _mm_add_pd(_mm_sub_pd(x_mul_x, y_mul_y), vec_x0);

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

        repeats[0]++;
        repeats[1]++;

        if(!(repeats[0] < iters && vec_length_squared[0] < 4)) {
            done[0] = 1;
        }

        if(!(repeats[1] < iters && vec_length_squared[1] < 4)) {
            done[1] = 1;
        }

        if(done[0] == 1) {
            done[0] = 0;
            if(i0 < width) {
                image[j * width + i0] = repeats[0];
            }
            i++;
            i0 = i;
            vec_x0[0] = i0 * x0_0 + left;
            vec_length_squared[0] = 0;
            repeats[0] = 0;
            vec_x[0] = 0;
            vec_y[0] = 0;
            x_mul_x[0] = vec_x[0] * vec_x[0];
            y_mul_y[0] = vec_y[0] * vec_y[0];
        }

        if(done[1] == 1) {
            done[1] = 0;
            if(i1 < width) {
                image[j * width + i1] = repeats[1];
            }
            i++;
            i1 = i;
            vec_x0[1] = i1 * x0_0 + left;
            vec_length_squared[1] = 0;
            repeats[1] = 0;
            vec_x[1] = 0;
            vec_y[1] = 0;
            x_mul_x[1] = vec_x[1] * vec_x[1];
            y_mul_y[1] = vec_y[1] * vec_y[1];
        }
    }
}

void* calculate(void* threadid) {

    int cur_row;
    int cur_start, cur_end;

    while(1) {
        // get row
        pthread_mutex_lock(&mutex);
        cur_height++;
        cur_row = cur_height;
        pthread_mutex_unlock(&mutex);

        if(cur_row >= height) {
            break;
        }

        cal_row(cur_row);      
    }

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

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

    /* init data & pthread */
	ncpus = CPU_COUNT(&cpu_set);

    int num_threads = ncpus;
    pthread_t threads[num_threads];
    int rc;
    int ID[num_threads];
    int t;

    y0_0 = ((upper - lower) / height);
    x0_0 = ((right - left) / width);

    pthread_mutex_init(&mutex, NULL);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* create pthread to solve mandelbrot set */
    for (t = 0; t < num_threads; t++) {
        ID[t] = t;
        rc = pthread_create(&threads[t], NULL, calculate, (void*)&ID[t]);
    }

    /* join pthread */
    for (int i = 0; i < num_threads; ++i)
		pthread_join(threads[i], NULL);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    pthread_exit(NULL);
}
