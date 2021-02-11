#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

double forward_2D_constant_density(float *grid, float *vel_base,
                                   float *damp, float *wavelet,
                                   size_t *src_points_interval, float *src_points_values,
                                   size_t *rec_points_interval, float *rec_points_values,
                                   float *receivers, size_t num_sources, size_t num_receivers,
                                   size_t nz, size_t nx, float dz, float dx,
                                   size_t jumps, float dt,
                                   size_t begin_timestep, size_t end_timestep, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0, nominator = 0.0;
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
    for(size_t n = begin_timestep; n < end_timestep; n++) {

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */

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

                //nominator with damp coefficient
                nominator = (1.0 + damp[current] * dt);

                value *= (dtSquared * vel_base[current] * vel_base[current]) / nominator;

                next_base[current] = 2.0 / nominator * prev_base[current] - ((1.0 - damp[current] * dt) / nominator) * next_base[current] + value;
            }
        }

        /*
            Section 2: add the source term
        */

        // for each source
        for(size_t src = 0; src < num_sources; src++){

            // each source has 4 (z_b, z_e, x_b, x_e) point intervals
            size_t offset_src = src * 4;

            // interval of grid points of the source in the Z axis
            size_t src_z_begin = src_points_interval[offset_src + 0];
            size_t src_z_end = src_points_interval[offset_src + 1];

            // interval of grid points of the source in the X axis
            size_t src_x_begin = src_points_interval[offset_src + 2];
            size_t src_x_end = src_points_interval[offset_src + 3];

            // number of grid points of the source in each axis
            size_t src_z_num_points = src_z_end - src_z_begin + 1;
            size_t src_x_num_points = src_x_end - src_x_begin + 1;

            // index of the Kaiser windowed sinc value of the source point
            offset_src = src * (src_z_num_points + src_x_num_points);
            size_t kws_index_z = offset_src;
            size_t kws_index_x = offset_src + src_z_num_points;

            // for each source point in the Z axis
            for(size_t i = src_z_begin; i <= src_z_end; i++){
                kws_index_x = offset_src + src_z_num_points;

                // for each source point in the X axis
                for(size_t j = src_x_begin; j <= src_x_end; j++){

                    float kws = src_points_values[kws_index_z] * src_points_values[kws_index_x];

                    // current source point in the grid
                    current = i * nx + j;
                    next_base[current] += dtSquared * vel_base[current] * vel_base[current] * kws * wavelet[n];

                    kws_index_x++;
                }
                kws_index_z++;
            }
        }

        /*
            Section 3: add null dirichlet (left, right, bottom) and null neumann (top) boundary conditions
        */

        // dirichlet on the left and right
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){
            // dirichlet on the left (first column)
            current = i * nx + 1;
            next_base[current] = 0.0;

            // dirichlet on the right (last column)
            current = i * nx + (nx - stencil_radius - 1);
            next_base[current] = 0.0;
        }

        // dirichlet on the bottom and neumann on the top
        for(size_t j = stencil_radius; j < nx - stencil_radius; j++){
            // dirichlet on the bottom (last row)
            current = (nz - stencil_radius - 1) * nx + j;
            next_base[current] = 0.0;

            // neumann on the top (top row in halo zone)
            size_t top_halo = 0 * nx + j;
            current = 2 * nx + j;
            next_base[top_halo] = next_base[current];
        }

        /*
            Section 4: compute the receivers
        */

        // for each receiver
        for(size_t rec = 0; rec < num_receivers; rec++){

            float sum = 0.0;

            // each receiver has 4 (z_b, z_e, x_b, x_e) point intervals
            size_t offset_rec = rec * 4;

            // interval of grid points of the receiver in the Z axis
            size_t rec_z_begin = rec_points_interval[offset_rec + 0];
            size_t rec_z_end = rec_points_interval[offset_rec + 1];

            // interval of grid points of the receiver in the X axis
            size_t rec_x_begin = rec_points_interval[offset_rec + 2];
            size_t rec_x_end = rec_points_interval[offset_rec + 3];

            // number of grid points of the receiver in each axis
            size_t rec_z_num_points = rec_z_end - rec_z_begin + 1;
            size_t rec_x_num_points = rec_x_end - rec_x_begin + 1;

            // index of the Kaiser windowed sinc value of the receiver point
            offset_rec = rec * (rec_z_num_points + rec_x_num_points);
            size_t kws_index_z = offset_rec;
            size_t kws_index_x = offset_rec + rec_z_num_points;

            // for each receiver point in the Z axis
            for(size_t i = rec_z_begin; i <= rec_z_end; i++){
                kws_index_x = offset_rec + rec_z_num_points;

                // for each receiver point in the X axis
                for(size_t j = rec_x_begin; j <= rec_x_end; j++){

                    float kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x];

                    // current receiver point in the grid
                    current = i * nx + j;
                    sum += prev_base[current] * kws;

                    kws_index_x++;
                }
                kws_index_z++;
            }

            size_t current_rec_n = n * num_receivers + rec;
            receivers[current_rec_n] = sum;
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        // save wavefield
        /*
        for(size_t i = 0; i < nz; i++){
            size_t offset_local = i * nx;
            size_t offset_global = (n * nz + i) * nx;

            for(size_t j = 0; j < nx; j++){
                grid[offset_global + j] = next_base[offset_local + j];
            }
        }*/
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

double forward_3D_constant_density(float *grid, float *vel_base,
                                   float *damp, float *wavelet,
                                   size_t *src_points_interval, float *src_points_values,
                                   size_t *rec_points_interval, float *rec_points_values,
                                   float *receivers, size_t num_sources, size_t num_receivers,
                                   size_t nz, size_t nx, size_t ny, float dz, float dx, float dy,
                                   size_t jumps, float dt,
                                   size_t begin_timestep, size_t end_timestep, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0, nominator = 0.0;
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
    for(size_t n = begin_timestep; n < end_timestep; n++) {

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */

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

                    //nominator with damp coefficient
                    nominator = (1.0 + damp[current] * dt);

                    value *= (dtSquared * vel_base[current] * vel_base[current]) / nominator;

                    next_base[current] = 2.0 / nominator * prev_base[current] - ((1.0 - damp[current] * dt) / nominator) * next_base[current] + value;
                }
            }
        }

        /*
            Section 2: add the source term
        */

        // for each source
        for(size_t src = 0; src < num_sources; src++){

            // each source has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
            size_t offset_src = src * 6;

            // interval of grid points of the source in the Z axis
            size_t src_z_begin = src_points_interval[offset_src + 0];
            size_t src_z_end = src_points_interval[offset_src + 1];

            // interval of grid points of the source in the X axis
            size_t src_x_begin = src_points_interval[offset_src + 2];
            size_t src_x_end = src_points_interval[offset_src + 3];

            // interval of grid points of the source in the Y axis
            size_t src_y_begin = src_points_interval[offset_src + 4];
            size_t src_y_end = src_points_interval[offset_src + 5];

            // number of grid points of the source in each axis
            size_t src_z_num_points = src_z_end - src_z_begin + 1;
            size_t src_x_num_points = src_x_end - src_x_begin + 1;
            size_t src_y_num_points = src_y_end - src_y_begin + 1;

            // index of the Kaiser windowed sinc value of the source point
            offset_src = src * (src_z_num_points + src_x_num_points + src_y_num_points);
            size_t kws_index_z = offset_src;
            size_t kws_index_x = offset_src + src_z_num_points;
            size_t kws_index_y = offset_src + src_z_num_points + src_x_num_points;

            // for each source point in the Z axis
            for(size_t i = src_z_begin; i <= src_z_end; i++){
                kws_index_x = offset_src + src_z_num_points;

                // for each source point in the X axis
                for(size_t j = src_x_begin; j <= src_x_end; j++){

                    kws_index_y = offset_src + src_z_num_points + src_x_num_points;

                    // for each source point in the Y axis
                    for(size_t k = src_y_begin; k <= src_y_end; k++){

                        float kws = src_points_values[kws_index_z] * src_points_values[kws_index_x] * src_points_values[kws_index_y];

                        // current source point in the grid
                        current = (i * nx + j) * ny + k;
                        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * kws * wavelet[n];

                        kws_index_y++;
                    }
                    kws_index_x++;
                }
                kws_index_z++;
            }
        }

        /*
            Section 3: add null dirichlet (left, right, back, front, bottom) and null neumann (top) boundary conditions
        */

        // dirichlet on the left and right (fixed on Y)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++){
                // dirichlet on the left (first column)
                current = (i * nx + j) * ny + 1;
                next_base[current] = 0.0;

                // dirichlet on the right (last column)
                current = (i * nx + j) * ny + (ny - stencil_radius - 1);
                next_base[current] = 0.0;
            }
        }

        // dirichlet on the back and front (fixed on X)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){
            for(size_t k = stencil_radius; k < ny - stencil_radius; k++){
                // dirichlet on the back (first column)
                current = (i * nx + 1) * ny + k;
                next_base[current] = 0.0;

                // dirichlet on the front (last column)
                current = (i * nx + (nx - stencil_radius - 1)) * ny + k;
                next_base[current] = 0.0;
            }
        }

        // dirichlet on the bottom and neumann on the top (fixed on Z)
        for(size_t j = stencil_radius; j < nx - stencil_radius; j++){
            for(size_t k = stencil_radius; k < ny - stencil_radius; k++){
                // dirichlet on the bottom (last row)
                current = ((nz - stencil_radius - 1) * nx + j) * ny + k;
                next_base[current] = 0.0;

                // neumann on the top (top row in halo zone)
                size_t top_halo = (0 * nx + j) * ny + k;
                current = (2 * nx + j) * ny + k;
                next_base[top_halo] = next_base[current];
            }
        }

        /*
            Section 4: compute the receivers
        */

        // for each receiver
        for(size_t rec = 0; rec < num_receivers; rec++){

            float sum = 0.0;

            // each receiver has 6 (z_b, z_e, x_b, x_e, y_b, y_e) point intervals
            size_t offset_rec = rec * 6;

            // interval of grid points of the receiver in the Z axis
            size_t rec_z_begin = rec_points_interval[offset_rec + 0];
            size_t rec_z_end = rec_points_interval[offset_rec + 1];

            // interval of grid points of the receiver in the X axis
            size_t rec_x_begin = rec_points_interval[offset_rec + 2];
            size_t rec_x_end = rec_points_interval[offset_rec + 3];

            // interval of grid points of the receiver in the Y axis
            size_t rec_y_begin = rec_points_interval[offset_rec + 4];
            size_t rec_y_end = rec_points_interval[offset_rec + 5];

            // number of grid points of the receiver in each axis
            size_t rec_z_num_points = rec_z_end - rec_z_begin + 1;
            size_t rec_x_num_points = rec_x_end - rec_x_begin + 1;
            size_t rec_y_num_points = rec_y_end - rec_y_begin + 1;

            // index of the Kaiser windowed sinc value of the receiver point
            offset_rec = rec * (rec_z_num_points + rec_x_num_points + rec_y_num_points);
            size_t kws_index_z = offset_rec;
            size_t kws_index_x = offset_rec + rec_z_num_points;
            size_t kws_index_y = offset_rec + rec_z_num_points + rec_x_num_points;

            // for each receiver point in the Z axis
            for(size_t i = rec_z_begin; i <= rec_z_end; i++){
                kws_index_x = offset_rec + rec_z_num_points;

                // for each receiver point in the X axis
                for(size_t j = rec_x_begin; j <= rec_x_end; j++){

                    kws_index_y = offset_rec + rec_z_num_points + rec_x_num_points;

                    // for each source point in the Y axis
                    for(size_t k = rec_y_begin; k <= rec_y_end; k++){

                        float kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x] * rec_points_values[kws_index_y];

                        // current receiver point in the grid
                        current = (i * nx + j) * ny + k;
                        sum += prev_base[current] * kws;

                        kws_index_y++;
                    }
                    kws_index_x++;
                }
                kws_index_z++;
            }

            size_t current_rec_n = n * num_receivers + rec;
            receivers[current_rec_n] = sum;
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        // save wavefield
        /*
        for(size_t i = 0; i < nz; i++){
            size_t offset_local = i * nx;
            size_t offset_global = (n * nz + i) * nx;

            for(size_t j = 0; j < nx; j++){
                grid[offset_global + j] = next_base[offset_local + j];
            }
        }*/
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

/*
double forward_2D_variable_density(float *grid, float *vel_base, float *density, float *damp,
                                   size_t nz, size_t nx, float dz, float dx,
                                   float *src, size_t origin_z, size_t origin_x,
                                   size_t timesteps, float dt,
                                   float *coeff, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0, nominator = 0.0;
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

                //nominator with damp coefficient
                nominator = (1.0 + damp[current] * dt);

                value = dtSquared * vel_base[current] * vel_base[current] * (term_z + term_x) / nominator;
                next_base[current] = 2.0 / nominator * prev_base[current] - ((1.0 - damp[current] * dt) / nominator) * next_base[current] + value;
            }
        }

        // add source term
        current = origin_z * nx + origin_x;
        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];

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


double forward_3D_constant_density(float *grid, float *vel_base,
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

        // add source term
        current = (origin_z * nx + origin_x) * ny + origin_y;
        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];

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

double forward_3D_variable_density(float *grid, float *vel_base, float *density,
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

                }
            }
        }

        // add source term
        current = (origin_z * nx + origin_x) * ny + origin_y;
        next_base[current] += dtSquared * vel_base[current] * vel_base[current] * src[n];

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
*/
