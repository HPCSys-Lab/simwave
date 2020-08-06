#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define HALF_LENGTH 1 // radius of the stencil

/*
 * save the matrix on a file.txt
 */
void save_grid(int iteration, int rows, int cols, float *matrix){

    system("mkdir -p wavefield");

    char file_name[256];
    sprintf(file_name, "wavefield/wavefield-iter-%d-grid-%d-%d.txt", iteration, rows, cols);

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < rows; i++) {

        int offset = i * cols;

        for(int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[offset + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int acoustic_forward_old(float *grid, float *vel_base, size_t nx, size_t ny, size_t nt, float dx, float dy, float dt, int print_every){

    // number of rows of the grid
    size_t rows = nx;

    // number of columns of the grid
    size_t cols = ny;

    // number of timesteps
    size_t iterations = nt;

    float *swap;
    float value = 0.0;
    int current;

    float dxSquared = dx * dx;
    float dySquared = dy * dy;
    float dtSquared = dt * dt;

    float *prev_base = malloc(rows * cols * sizeof(float));
    float *next_base = malloc(rows * cols * sizeof(float));

    // initialize aux matrix
    for(size_t i = 0; i < rows; i++){

        size_t offset = i * cols;

        for(size_t j = 0; j < cols; j++){
            prev_base[offset + j] = grid[offset + j];
            next_base[offset + j] = 0.0f;
        }
    }

    // wavefield modeling
    for(size_t n = 0; n < iterations; n++) {
        for(size_t i = 1; i < rows - HALF_LENGTH; i++) {
            for(size_t j = 1; j < cols - HALF_LENGTH; j++) {
                // index of the current point in the grid
                current = i * cols + j;

                // stencil code to update grid
                value = 0.0;
                //neighbors in the horizontal direction
                value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;
                //neighbors in the vertical direction
                value += (prev_base[current + cols] - 2.0 * prev_base[current] + prev_base[current - cols]) / dySquared;
                value *= dtSquared * vel_base[current] * vel_base[current];
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
            }
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        if (print_every && n % print_every == 0 )
            save_grid(n, rows, cols, next_base);
    }

    // save final result
    for(size_t i = 0; i < rows; i++){

        size_t offset = i * cols;

        for(size_t j = 0; j < cols; j++){
            grid[offset + j] = next_base[offset + j];
        }
    }

    free(prev_base);
    free(next_base);

    return 0;

}

// ======= Nova implementacao
int acoustic_forward(float *grid, float *vel_base, float *density, size_t nx, size_t ny, size_t nt, float dx, float dy, float dt, int print_every){

    // number of rows of the grid
    size_t rows = nx;

    // number of columns of the grid
    size_t cols = ny;

    // number of timesteps
    size_t iterations = nt;

    float *swap;
    float value = 0.0;
    int current;

    float dxSquared = dx * dx;
    float dySquared = dy * dy;
    float dtSquared = dt * dt;

    float *prev_base = malloc(rows * cols * sizeof(float));
    float *next_base = malloc(rows * cols * sizeof(float));

    // initialize aux matrix
    for(size_t i = 0; i < rows; i++){

        size_t offset = i * cols;

        for(size_t j = 0; j < cols; j++){
            prev_base[offset + j] = grid[offset + j];
            next_base[offset + j] = 0.0f;
        }
    }

    // wavefield modeling
    for(size_t n = 0; n < iterations; n++) {
        for(size_t i = 1; i < rows - HALF_LENGTH; i++) {
            for(size_t j = 1; j < cols - HALF_LENGTH; j++) {
                // index of the current point in the grid
                current = i * cols + j;

                // stencil code to update grid
                //value = 0.0;
                //neighbors in the horizontal direction
                //value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;
                float x1 = ((prev_base[current + 1] - prev_base[current]) * (density[current + 1] + density[current])) / density[current + 1];
                float x2 = ((prev_base[current] - prev_base[current - 1]) * (density[current] + density[current - 1])) / density[current - 1];
                float termo_x = (x1 - x2) / (2 * dxSquared);
                //neighbors in the vertical direction
//                value += (prev_base[current + cols] - 2.0 * prev_base[current] + prev_base[current - cols]) / dySquared;
                float y1 = ((prev_base[current + cols] - prev_base[current]) * (density[current + cols] + density[current])) / density[current + cols];
                float y2 = ((prev_base[current] - prev_base[current - cols]) * (density[current] + density[current - cols])) / density[current - cols];
                float termo_y = (y1 - y2) / (2 * dySquared);

//                value *= dtSquared * vel_base[current];
                value = dtSquared * vel_base[current] * (termo_x - termo_y);
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
            }
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        if (print_every && n % print_every == 0 )
            save_grid(n, rows, cols, next_base);
    }

    // save final result
    for(size_t i = 0; i < rows; i++){

        size_t offset = i * cols;

        for(size_t j = 0; j < cols; j++){
            grid[offset + j] = next_base[offset + j];
        }
    }

    free(prev_base);
    free(next_base);

    return 0;

}
