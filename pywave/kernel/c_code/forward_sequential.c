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

// MULTI SPATIAL ORDER VERSION
double forward_2D_constant_density(float *grid, float *vel_base, float *src, size_t origin_z, size_t origin_x, size_t nz, size_t nx, size_t timesteps, float dz, float dx, float dt, int print_every){

    size_t spatial_order = 8;
    size_t stencil_radius = spatial_order / 2;

    // array of coefficients
    float coefficient[stencil_radius + 1];

    // get the coefficients for the specific spatial ordem
    switch (spatial_order){
        case 2:
            coefficient[0] = -2.0;
            coefficient[1] = 1.0;
        break;

        case 4:
            coefficient[0] = -2.50000e+0;
            coefficient[1] = 1.33333e+0;
            coefficient[2] = -8.33333e-2;
        break;

        case 6:
            coefficient[0] = -2.72222e+0;
            coefficient[1] = 1.50000e+0;
            coefficient[2] = -1.50000e-1;
            coefficient[3] = 1.11111e-2;
        break;

        case 8:
            coefficient[0] = -2.84722e+0;
            coefficient[1] = 1.60000e+0;
            coefficient[2] = -2.00000e-1;
            coefficient[3] = 2.53968e-2;
            coefficient[4] = -1.78571e-3;
        break;

        case 10:
            coefficient[0] = -2.92722e+0;
            coefficient[1] = 1.66667e+0;
            coefficient[2] = -2.38095e-1;
            coefficient[3] = 3.96825e-2;
            coefficient[4] = -4.96032e-3;
            coefficient[5] = 3.17460e-4;
        break;

        case 12:
            coefficient[0] = -2.98278e+0;
            coefficient[1] = 1.71429e+0;
            coefficient[2] = -2.67857e-1;
            coefficient[3] = 5.29101e-2;
            coefficient[4] = -8.92857e-3;
            coefficient[5] = 1.03896e-3;
            coefficient[6] = -6.01251e-5;
        break;

        case 14:
            coefficient[0] = -3.02359e+0;
            coefficient[1] = 1.75000e+0;
            coefficient[2] = -2.91667e-1;
            coefficient[3] = 6.48148e-2;
            coefficient[4] = -1.32576e-2;
            coefficient[5] = 2.12121e-3;
            coefficient[6] = -2.26625e-4;
            coefficient[7] = 1.18929e-5;
        break;

        case 16:
            coefficient[0] = -3.05484e+0;
            coefficient[1] = 1.77778e+0;
            coefficient[2] = -3.11111e-1;
            coefficient[3] = 7.54209e-2;
            coefficient[4] = -1.76768e-2;
            coefficient[5] = 3.48096e-3;
            coefficient[6] = -5.18001e-4;
            coefficient[7] = 5.07429e-5;
            coefficient[8] = -2.42813e-6;
        break;
    }

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
                value += coefficient[0] * (prev_base[current]/dxSquared + prev_base[current]/dzSquared);

                // radius of the stencil
                for(int ir = 1; ir <= stencil_radius; ir++){
                    value += coefficient[ir] * (
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

        if (print_every && n % print_every == 0 )
            save_grid(n, nz, nx, next_base);
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

/*
double forward_2D_constant_density(float *grid, float *vel_base, float *src, size_t origin_z, size_t origin_x, size_t nz, size_t nx, size_t timesteps, float dz, float dx, float dt, int print_every){

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
        for(size_t i = 1; i < nz - HALF_LENGTH; i++) {
            for(size_t j = 1; j < nx - HALF_LENGTH; j++) {
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

        if (print_every && n % print_every == 0 )
            save_grid(n, nz, nx, next_base);
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
*/

double forward_2D_variable_density(float *grid, float *vel_base, float *density, size_t nz, size_t nx, size_t timesteps, float dz, float dx, float dt, int print_every){

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

    float z1, z2, x1, x2, term_z, term_x;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        for(size_t i = 1; i < nz - HALF_LENGTH; i++) {
            for(size_t j = 1; j < nx - HALF_LENGTH; j++) {
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
            }
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        if (print_every && n % print_every == 0 )
            save_grid(n, nz, nx, next_base);
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

double forward_3D_constant_density(float *grid, float *vel_base, size_t nz, size_t nx, size_t ny, size_t timesteps, float dz, float dx, float dy, float dt, int print_every){

    float *swap;
    float value = 0.0;
    size_t current;

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
        for(size_t i = 1; i < nz - HALF_LENGTH; i++) {
            for(size_t j = 1; j < nx - HALF_LENGTH; j++) {
                for(size_t k = 1; k < ny - HALF_LENGTH; k++) {
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

        if (print_every && n % print_every == 0 )
            save_grid(n, nz, nx, next_base);
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

double forward_3D_variable_density(float *grid, float *vel_base, float *density, size_t nz, size_t nx, size_t ny, size_t timesteps, float dz, float dx, float dy, float dt, int print_every){

    float *swap;
    float value = 0.0;
    size_t current;

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

    float z1, z2, x1, x2, y1, y2, term_z, term_x, term_y;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(size_t n = 0; n < timesteps; n++) {
        for(size_t i = 1; i < nz - HALF_LENGTH; i++) {
            for(size_t j = 1; j < nx - HALF_LENGTH; j++) {
                for(size_t k = 1; k < ny - HALF_LENGTH; k++) {
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

                }
            }
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        if (print_every && n % print_every == 0 )
            save_grid(n, nz, nx, next_base);
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
