#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
//#include <omp.h>

double lean_forward_3D(float *grid, float *vel_base,
                       size_t nz, size_t nx, size_t ny,
                       float dz, float dx, float dy,
                       float dt, size_t timesteps, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0;
    int current;

    float dzSquared = dz * dz;
    float dxSquared = dx * dx;
    float dySquared = dy * dy;
    float dtSquared = dt * dt;

    size_t nsize = nz * nx * ny;

    float *prev_base = malloc(nsize * sizeof(float));
    float *next_base = malloc(nsize * sizeof(float));

    // initialize aux matrix
    for(size_t i = 0; i < nz; i++){
        for(size_t j = 0; j < nx; j++){

            size_t offset = (i * nx + j) * ny;

            for(size_t k = 0; k < ny; k++){
                prev_base[offset + k] = grid[offset + k];
                next_base[offset + k] = 0.0f;
            }
        }
    }

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        #pragma omp parallel for private(current, value)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    current = (i * nx + j) * ny + k;

                    // stencil code to update grid
                    value = 0.0;

                    //neighbors in the Y direction
                    value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dySquared;

                    //neighbors in the X direction
                    value += (prev_base[current + ny] - 2.0 * prev_base[current] + prev_base[current - ny]) / dxSquared;

                    //neighbors in the Z direction
                    value += (prev_base[current + (nx * ny)] - 2.0 * prev_base[current] + prev_base[current - (nx * ny)]) / dzSquared;

                    value *= dtSquared * vel_base[current] * vel_base[current];

                    next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
                }
            }
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    // save final result
    for(size_t i = 0; i < nz; i++){
        for(size_t j = 0; j < nx; j++){
            size_t offset = (i * nx + j) * ny;

            for(size_t k = 0; k < ny; k++){
                grid[offset + k] = next_base[offset + k];
            }
        }
    }

    free(prev_base);
    free(next_base);

    return exec_time;
}
