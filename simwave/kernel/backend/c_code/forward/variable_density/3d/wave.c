#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

// forward_3D_variable_density
double forward(f_type *grid, f_type *velocity, f_type *density, f_type *damp,
               f_type *wavelet, f_type *coeff, size_t *boundary_conditions,
               size_t *src_points_interval, f_type *src_points_values,
               size_t *rec_points_interval, f_type *rec_points_values,
               f_type *receivers, size_t num_sources, size_t num_receivers,
               size_t nz, size_t nx, size_t ny, f_type dz, f_type dx, f_type dy,
               size_t saving_stride, f_type dt,
               size_t begin_timestep, size_t end_timestep, size_t space_order){

    size_t stencil_radius = space_order / 2;

    f_type *swap;
    size_t current;
    size_t wavefield_count = 0;

    f_type dzSquared = dz * dz;
    f_type dxSquared = dx * dx;
    f_type dySquared = dy * dy;
    f_type dtSquared = dt * dt;

    size_t nsize = nz * nx * ny;

    f_type *prev_snapshot = malloc(nsize * sizeof(f_type));
    f_type *next_snapshot = malloc(nsize * sizeof(f_type));

    // initialize aux matrix
    for(size_t i = 0; i < nz; i++){
        for(size_t j = 0; j < nx; j++){

            size_t offset = (i * nx + j) * ny;

            for(size_t k = 0; k < ny; k++){
                prev_snapshot[offset + k] = grid[offset + k];
                next_snapshot[offset + k] = 0.0f;
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

                    //neighbors in the Y direction
                    f_type y1 = ((prev_snapshot[current + 1] - prev_snapshot[current]) * (density[current + 1] + density[current])) / density[current + 1];
                    f_type y2 = ((prev_snapshot[current] - prev_snapshot[current - 1]) * (density[current] + density[current - 1])) / density[current - 1];
                    f_type term_y = (y1 - y2) / (2 * dySquared);

                    //neighbors in the X direction
                    f_type x1 = ((prev_snapshot[current + ny] - prev_snapshot[current]) * (density[current + ny] + density[current])) / density[current + ny];
                    f_type x2 = ((prev_snapshot[current] - prev_snapshot[current - ny]) * (density[current] + density[current - ny])) / density[current - ny];
                    f_type term_x = (x1 - x2) / (2 * dxSquared);

                    //neighbors in the Z direction
                    f_type z1 = ((prev_snapshot[current + (nx * ny)] - prev_snapshot[current]) * (density[current + (nx * ny)] + density[current])) / density[current + (nx * ny)];
                    f_type z2 = ((prev_snapshot[current] - prev_snapshot[current - (nx * ny)]) * (density[current] + density[current - (nx * ny)])) / density[current - (nx * ny)];
                    f_type term_z = (z1 - z2) / (2 * dzSquared);

                    //nominator with damp coefficient
                    f_type denominator = (1.0 + damp[current] * dt);

                    f_type value = dtSquared * velocity[current] * velocity[current] * (term_z + term_x + term_y) / denominator;

                    next_snapshot[current] = 2.0 / denominator * prev_snapshot[current] - ((1.0 - damp[current] * dt) / denominator) * next_snapshot[current] + value;

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

                        f_type kws = src_points_values[kws_index_z] * src_points_values[kws_index_x] * src_points_values[kws_index_y];

                        // current source point in the grid
                        current = (i * nx + j) * ny + k;
                        next_snapshot[current] += dtSquared * velocity[current] * velocity[current] * kws * wavelet[n];

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
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the left
                if(y_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + j) * ny + stencil_radius;
                        next_snapshot[current - ir] = next_snapshot[current + ir];
                    }
                }

                // null dirichlet on the right
                if(y_after == 1){
                    current = (i * nx + j) * ny + (ny - stencil_radius - 1);
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the right
                if(y_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + j) * ny + (ny - stencil_radius - 1);
                        next_snapshot[current + ir] = next_snapshot[current - ir];
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
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the front
                if(x_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + stencil_radius) * ny + k;
                        next_snapshot[current - (ir * ny)] = next_snapshot[current + (ir * ny)];
                    }
                }

                // null dirichlet on the back
                if(x_after == 1){
                    current = (i * nx + (nx - stencil_radius - 1)) * ny + k;
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the back
                if(x_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (i * nx + (nx - stencil_radius - 1)) * ny + k;
                        next_snapshot[current + (ir * ny)] = next_snapshot[current - (ir * ny)];
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
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the top
                if(z_before == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = (stencil_radius * nx + j) * ny + k;
                        next_snapshot[current - (ir * nx * ny)] = next_snapshot[current + (ir * nx * ny)];
                    }
                }

                // null dirichlet on the bottom
                if(z_after == 1){
                    current = ((nz - stencil_radius - 1) * nx + j) * ny + k;
                    next_snapshot[current] = 0.0;
                }

                // null neumann on the bottom
                if(z_after == 2){
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        current = ((nz - stencil_radius - 1) * nx + j) * ny + k;
                        next_snapshot[current + (ir * nx * ny)] = next_snapshot[current - (ir * nx * ny)];
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

            f_type sum = 0.0;

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

                        f_type kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x] * rec_points_values[kws_index_y];

                        // current receiver point in the grid
                        current = (i * nx + j) * ny + k;
                        sum += prev_snapshot[current] * kws;

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
        swap = next_snapshot;
        next_snapshot = prev_snapshot;
        prev_snapshot = swap;

        /*
            Section 5: save the wavefields
        */
        if( (saving_stride && (n % saving_stride) == 0) || (n == end_timestep - 1) ){

            for(size_t i = 0; i < nz; i++){
                for(size_t j = 0; j < nx; j++){

                    size_t offset_local = (i * nx + j) * ny;
                    size_t offset_global = ((wavefield_count * nz + i) * nx + j) * ny;

                    for(size_t k = 0; k < ny; k++){
                        grid[offset_global + k] = next_snapshot[offset_local + k];
                    }
                }
            }

            wavefield_count++;
        }

    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    free(prev_snapshot);
    free(next_snapshot);

    return exec_time;
}
