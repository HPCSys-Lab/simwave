#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// forward_3D_constant_density
double forward(float *grid, float *vel_base, float *damp,
               float *wavelet, float *coeff, size_t *boundary_conditions,
               size_t *src_points_interval, float *src_points_values,
               size_t *rec_points_interval, float *rec_points_values,
               float *receivers, size_t num_sources, size_t num_receivers,
               size_t nz, size_t nx, size_t ny, float dz, float dx, float dy,
               size_t jumps, float dt,
               size_t begin_timestep, size_t end_timestep, size_t space_order){

    size_t stencil_radius = space_order / 2;

    float *swap;
    float value = 0.0, nominator = 0.0;
    size_t current;
    size_t wavefield_count = 0;

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

        // pointer to src value offset
        size_t offset_src_kws_index_z = 0;

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
            size_t kws_index_z = offset_src_kws_index_z;

            // for each source point in the Z axis
            for(size_t i = src_z_begin; i <= src_z_end; i++){
                size_t kws_index_x = offset_src_kws_index_z + src_z_num_points;

                // for each source point in the X axis
                for(size_t j = src_x_begin; j <= src_x_end; j++){

                    size_t kws_index_y = offset_src_kws_index_z + src_z_num_points + src_x_num_points;

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

            offset_src_kws_index_z += (src_z_num_points + src_x_num_points + src_y_num_points);
        }

        /*
            Section 3: add boundary conditions (z_before, z_after, x_before, x_after, y_before, y_after)
            0 - no boundary condition
            1 - null dirichlet
            2 - null neumann
        */
        size_t z_before = boundary_conditions[0];
        size_t z_after = boundary_conditions[1];
        size_t x_before = boundary_conditions[2];
        size_t x_after = boundary_conditions[3];
        size_t y_before = boundary_conditions[4];
        size_t y_after = boundary_conditions[5];

        // boundary conditions on the left and right (fixed on Y)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++){

                // null dirichlet on the left
                if(y_before == 1){
                    current = (i * nx + j) * ny + stencil_radius;
                    next_base[current] = 0.0;
                }

                // null neumann on the left
                if(y_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + j) * ny + stencil_radius;
                        next_base[current - ir] = next_base[current + ir];
                    }
                }

                // null dirichlet on the right
                if(y_after == 1){
                    current = (i * nx + j) * ny + (ny - stencil_radius - 1);
                    next_base[current] = 0.0;
                }

                // null neumann on the right
                if(y_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + j) * ny + (ny - stencil_radius - 1);
                        next_base[current + ir] = next_base[current - ir];
                    }
                }

            }
        }

        // boundary conditions on the front and back (fixed on X)
        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){
            for(size_t k = stencil_radius; k < ny - stencil_radius; k++){

                // null dirichlet on the front
                if(x_before == 1){
                    current = (i * nx + stencil_radius) * ny + k;
                    next_base[current] = 0.0;
                }

                // null neumann on the front
                if(x_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + stencil_radius) * ny + k;
                        next_base[current - (ir * ny)] = next_base[current + (ir * ny)];
                    }
                }

                // null dirichlet on the back
                if(x_after == 1){
                    current = (i * nx + (nx - stencil_radius - 1)) * ny + k;
                    next_base[current] = 0.0;
                }

                // null neumann on the back
                if(x_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + (nx - stencil_radius - 1)) * ny + k;
                        next_base[current + (ir * ny)] = next_base[current - (ir * ny)];
                    }
                }

            }
        }

        // boundary conditions on the bottom and top (fixed on Z)
        for(size_t j = stencil_radius; j < nx - stencil_radius; j++){
            for(size_t k = stencil_radius; k < ny - stencil_radius; k++){

                // null dirichlet on the top
                if(z_before == 1){
                    current = (stencil_radius * nx + j) * ny + k;
                    next_base[current] = 0.0;
                }

                // null neumann on the top
                if(z_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (stencil_radius * nx + j) * ny + k;
                        next_base[current - (ir * nx * ny)] = next_base[current + (ir * nx * ny)];
                    }
                }

                // null dirichlet on the bottom
                if(z_after == 1){
                    current = ((nz - stencil_radius - 1) * nx + j) * ny + k;
                    next_base[current] = 0.0;
                }

                // null neumann on the bottom
                if(z_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = ((nz - stencil_radius - 1) * nx + j) * ny + k;
                        next_base[current + (ir * nx * ny)] = next_base[current - (ir * nx * ny)];
                    }
                }

            }
        }

        /*
            Section 4: compute the receivers
        */

        // pointer to rec value offset
        size_t offset_rec_kws_index_z = 0;

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
            size_t kws_index_z = offset_rec_kws_index_z;

            // for each receiver point in the Z axis
            for(size_t i = rec_z_begin; i <= rec_z_end; i++){
                size_t kws_index_x = offset_rec_kws_index_z + rec_z_num_points;

                // for each receiver point in the X axis
                for(size_t j = rec_x_begin; j <= rec_x_end; j++){

                    size_t kws_index_y = offset_rec_kws_index_z + rec_z_num_points + rec_x_num_points;

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

            offset_rec_kws_index_z += (rec_z_num_points + rec_x_num_points + rec_y_num_points);
        }

        //swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        /*
            Section 5: save the wavefields
        */
        if( (jumps && (n % jumps) == 0) || (n == end_timestep - 1) ){

            for(size_t i = 0; i < nz; i++){
                for(size_t j = 0; j < nx; j++){

                    size_t offset_local = (i * nx + j) * ny;
                    size_t offset_global = ((wavefield_count * nz + i) * nx + j) * ny;

                    for(size_t k = 0; k < ny; k++){
                        grid[offset_global + k] = next_base[offset_local + k];
                    }
                }
            }

            wavefield_count++;
        }

    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    free(prev_base);
    free(next_base);

    return exec_time;
}
