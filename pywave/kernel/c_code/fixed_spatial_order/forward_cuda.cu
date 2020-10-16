#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

extern "C" double forward_2D_constant_density(float *grid, float *vel_base,
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

    float *prev_base = (float*) malloc(nz * nx * sizeof(float));
    float *next_base = (float*) malloc(nz * nx * sizeof(float));

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

                //neighbors in the horizontal direction
                value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;

                //neighbors in the vertical direction
                value += (prev_base[current + nx] - 2.0 * prev_base[current] + prev_base[current - nx]) / dzSquared;

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


extern "C" double forward_2D_variable_density(float *grid, float *vel_base, float *density,
                                   size_t nz, size_t nx, float dz, float dx,
                                   float *src, size_t origin_z, size_t origin_x,
                                   size_t timesteps, float dt,
                                   float *coeff, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0;
    int current;

    float dzSquared = dz * dz;
    float dxSquared = dx * dx;
    float dtSquared = dt * dt;

    float *prev_base = (float*) malloc(nz * nx * sizeof(float));
    float *next_base = (float*) malloc(nz * nx * sizeof(float));

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

    float z1, z2, x1, x2, term_z, term_x;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                // index of the current point in the grid
                current = i * nx + j;

                //neighbors in the horizontal direction
                x1 = ((prev_base[current + 1] - prev_base[current]) * (density[current + 1] + density[current])) / density[current + 1];
                x2 = ((prev_base[current] - prev_base[current - 1]) * (density[current] + density[current - 1])) / density[current - 1];
                term_x = (x1 - x2) / (2 * dxSquared);

                //neighbors in the vertical direction
                z1 = ((prev_base[current + nx] - prev_base[current]) * (density[current + nx] + density[current])) / density[current + nx];
                z2 = ((prev_base[current] - prev_base[current - nx]) * (density[current] + density[current - nx])) / density[current - nx];
                term_z = (z1 - z2) / (2 * dzSquared);

                value = dtSquared * vel_base[current] * vel_base[current] * (term_z + term_x);
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


extern "C" double forward_3D_constant_density(float *grid, float *vel_base,
                                   size_t nz, size_t nx, size_t ny,
                                   float dz, float dx, float dy,
                                   float *src, size_t origin_z, size_t origin_x, size_t origin_y,
                                   size_t timesteps,  float dt,
                                   float *coeff, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0;
    size_t current;

    float dzSquared = dz * dz;
    float dxSquared = dx * dx;
    float dySquared = dy * dy;
    float dtSquared = dt * dt;

    size_t nsize = nz * nx * ny;

    float *prev_base = (float*) malloc(nsize * sizeof(float));
    float *next_base = (float*) malloc(nsize * sizeof(float));

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

                    if( i == origin_z && j == origin_x && k == origin_y)
                        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];
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

extern "C" double forward_3D_variable_density(float *grid, float *vel_base, float *density,
                                   size_t nz, size_t nx, size_t ny,
                                   float dz, float dx, float dy,
                                   float *src, size_t origin_z, size_t origin_x, size_t origin_y,
                                   size_t timesteps,  float dt,
                                   float *coeff, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0;
    size_t current;

    float dzSquared = dz * dz;
    float dxSquared = dx * dx;
    float dySquared = dy * dy;
    float dtSquared = dt * dt;

    size_t nsize = nz * nx * ny;

    float *prev_base = (float*) malloc(nsize * sizeof(float));
    float *next_base = (float*) malloc(nsize * sizeof(float));

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

    float z1, z2, x1, x2, y1, y2, term_z, term_x, term_y;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                for(size_t k = stencil_radius; k < ny - stencil_radius; k++) {
                    // index of the current point in the grid
                    current = (i * nx + j) * ny + k;

                    //neighbors in the Y direction
                    y1 = ((prev_base[current + 1] - prev_base[current]) * (density[current + 1] + density[current])) / density[current + 1];
                    y2 = ((prev_base[current] - prev_base[current - 1]) * (density[current] + density[current - 1])) / density[current - 1];
                    term_y = (y1 - y2) / (2 * dySquared);

                    //neighbors in the X direction
                    x1 = ((prev_base[current + ny] - prev_base[current]) * (density[current + ny] + density[current])) / density[current + ny];
                    x2 = ((prev_base[current] - prev_base[current - ny]) * (density[current] + density[current - ny])) / density[current - ny];
                    term_x = (x1 - x2) / (2 * dxSquared);

                    //neighbors in the Z direction
                    z1 = ((prev_base[current + (nx * ny)] - prev_base[current]) * (density[current + (nx * ny)] + density[current])) / density[current + (nx * ny)];
                    z2 = ((prev_base[current] - prev_base[current - (nx * ny)]) * (density[current] + density[current - (nx * ny)])) / density[current - (nx * ny)];
                    term_z = (z1 - z2) / (2 * dzSquared);

                    value = dtSquared * vel_base[current] * vel_base[current] * (term_z + term_x + term_y);
                    next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;

                    if( i == origin_z && j == origin_x && k == origin_y)
                        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];

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
