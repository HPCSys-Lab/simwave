#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

double forward_2D_constant_density(float *grid, float *vel_base,
                                   size_t nz, size_t nx, float dz, float dx,
                                   float *src, size_t origin_z, size_t origin_x,
                                   size_t timesteps,  float dt,
                                   float *coeff, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0;
    int current;

    float dzSquared = dz * dz;
    float dxSquared = dx * dx;
    float dtSquared = dt * dt;

    float *prev_base = malloc(nz * nx * sizeof(float));
    float *next_base = malloc(nz * nx * sizeof(float));

    // initialize aux matrix
    for(size_t i = 0; i < nz; i++){

        size_t offset = i * nx;

        for(size_t j = 0; j < nx; j++){
            prev_base[offset + j] = grid[offset + j];
            next_base[offset + j] = 0.0f;
        }
    }

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                // index of the current point in the grid
                current = i * nx + j;

                // stencil code to update grid
                value = 0.0;
                value += coeff[0] * (prev_base[current]/dxSquared + prev_base[current]/dzSquared);

                // radius of the stencil
                for(int ir = 1; ir <= stencil_radius; ir++){
                    value += coeff[ir] * (
                            ( (prev_base[current + ir] + prev_base[current - ir]) / dxSquared ) + //neighbors in the horizontal direction
                            ( (prev_base[current + (ir * nx)] + prev_base[current - (ir * nx)]) / dzSquared )); //neighbors in the vertical direction
                }

                value *= dtSquared * vel_base[current] * vel_base[current];
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;

                if( i == origin_z && j == origin_x )
                    next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];

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

        size_t offset = i * nx;

        for(size_t j = 0; j < nx; j++){
            grid[offset + j] = next_base[offset + j];
        }
    }

    free(prev_base);
    free(next_base);

    return exec_time;

}
