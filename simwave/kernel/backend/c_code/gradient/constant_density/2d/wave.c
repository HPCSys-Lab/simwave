#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#if defined(CPU_OPENMP) || defined(GPU_OPENMP)
    #include <omp.h>
#endif

#if defined(GPU_OPENACC)
    #include <openacc.h>
#endif

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

// gradient 2D_constant_density
double gradient(f_type *u, f_type *v, f_type *grad, f_type *velocity, f_type *damp,
               f_type *wavelet, size_t wavelet_size, size_t wavelet_count,
               f_type *coeff, size_t *boundary_conditions,
               size_t *src_points_interval, size_t src_points_interval_size,
               f_type *src_points_values, size_t src_points_values_size,
               size_t *src_points_values_offset,               
               size_t num_sources,
               size_t nz, size_t nx, f_type dz, f_type dx,
               size_t stride, f_type dt,
               size_t begin_timestep, size_t end_timestep,
               size_t space_order, size_t num_snapshots){

    size_t stencil_radius = space_order / 2;

    size_t domain_size = nz * nx;

    f_type dzSquared = dz * dz;
    f_type dxSquared = dx * dx;
    f_type dtSquared = dt * dt;

    // timestep pointers
    size_t prev_t = 0;
    size_t current_t = 1;
    size_t next_t = 2;

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    #ifdef GPU_OPENMP

    // select the device
    #ifdef DEVICEID
    omp_set_default_device(DEVICEID);
    #endif

    size_t u_size = num_snapshots * domain_size;
    size_t v_size = 3 * domain_size; // prev, current, next

    #pragma omp target enter data map(to: u[:u_size])
    #pragma omp target enter data map(to: v[:v_size])
    #pragma omp target enter data map(to: grad[:domain_size])
    #pragma omp target enter data map(to: velocity[:domain_size])
    #pragma omp target enter data map(to: damp[:domain_size])
    #pragma omp target enter data map(to: coeff[:stencil_radius+1])
    #pragma omp target enter data map(to: src_points_interval[:src_points_interval_size])
    #pragma omp target enter data map(to: src_points_values[:src_points_values_size])
    #pragma omp target enter data map(to: src_points_values_offset[:num_sources])    
    #pragma omp target enter data map(to: wavelet[:wavelet_size * wavelet_count])
   
    #endif

    #ifdef GPU_OPENACC

    // select the device
    #ifdef DEVICEID
    acc_init(acc_device_nvidia);
    acc_set_device_num(DEVICEID, acc_device_nvidia);
    #endif

    size_t u_size = num_snapshots * domain_size;
    size_t v_size = 3 * domain_size; // prev, current, next

    #pragma acc enter data copyin(u[:u_size])
    #pragma acc enter data copyin(v[:v_size])
    #pragma acc enter data copyin(grad[:domain_size])
    #pragma acc enter data copyin(velocity[:domain_size])
    #pragma acc enter data copyin(damp[:domain_size])
    #pragma acc enter data copyin(coeff[:stencil_radius+1])
    #pragma acc enter data copyin(src_points_interval[:src_points_interval_size])
    #pragma acc enter data copyin(src_points_values[:src_points_values_size])
    #pragma acc enter data copyin(src_points_values_offset[:num_sources])    
    #pragma acc enter data copyin(wavelet[:wavelet_size * wavelet_count])
    #endif
    
    // adjoint modeling   
    for(size_t n = end_timestep; n >= begin_timestep; n--) {

        // no saving case
        // since we are moving backwards, we invert the next_t and prev_t pointer       
        next_t = (n - 1) % 3;
        current_t = n % 3;
        prev_t = (n + 1) % 3;
        

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */

        #ifdef CPU_OPENMP
        #pragma omp parallel for simd
        #endif

        #ifdef GPU_OPENMP
        #pragma omp target teams distribute parallel for collapse(2)
        #endif

        #ifdef GPU_OPENACC
        #pragma acc parallel loop collapse(2) present(coeff,damp,v,velocity)
        #endif

        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                // index of the current point in the grid
                size_t domain_offset = i * nx + j;

                size_t prev_snapshot = prev_t * domain_size + domain_offset;
                size_t current_snapshot = current_t * domain_size + domain_offset;
                size_t next_snapshot = next_t * domain_size + domain_offset;

                // stencil code to update grid
                f_type value = 0.0;

                f_type sum_x = coeff[0] * v[current_snapshot];
                f_type sum_z = coeff[0] * v[current_snapshot];

                // radius of the stencil
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t ir = 1; ir <= stencil_radius; ir++){
                    //neighbors in the horizontal direction
                    sum_x += coeff[ir] * (v[current_snapshot + ir] + v[current_snapshot - ir]);

                    //neighbors in the vertical direction
                    sum_z += coeff[ir] * (v[current_snapshot + (ir * nx)] + v[current_snapshot - (ir * nx)]);
                }

                value += sum_x/dxSquared + sum_z/dzSquared;

                // parameter to be used
                f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                // denominator with damp coefficient
                f_type denominator = (1.0 + damp[domain_offset] * dt / 2);
                f_type numerator = (1.0 - damp[domain_offset] * dt / 2);

                value *= (dtSquared / slowness) / denominator;

                v[next_snapshot] = 2.0 / denominator * v[current_snapshot] - (numerator / denominator) * v[prev_snapshot] + value;
            }
        }

        /*
            Section 2: add the source term
            In this context, the sources are the same as the receivers from forward
        */

        #ifdef CPU_OPENMP
        #pragma omp parallel for
        #endif

        #ifdef GPU_OPENMP
        #pragma omp target teams distribute parallel for
        #endif

        #ifdef GPU_OPENACC
        #pragma acc parallel loop present(src_points_interval,src_points_values,src_points_values_offset,v,velocity,wavelet)
        #endif

        // for each source
        for(size_t src = 0; src < num_sources; src++){

            size_t wavelet_offset = n - 1;

            if(wavelet_count > 1){
                wavelet_offset = (n-1) * num_sources + src;
            }

            if(wavelet[wavelet_offset] != 0.0){

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
                //size_t src_x_num_points = src_x_end - src_x_begin + 1;

                // pointer to src value offset
                size_t offset_src_kws_index_z = src_points_values_offset[src];

                // index of the Kaiser windowed sinc value of the source point
                size_t kws_index_z = offset_src_kws_index_z;

                // for each source point in the Z axis
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t i = src_z_begin; i <= src_z_end; i++){
                    size_t kws_index_x = offset_src_kws_index_z + src_z_num_points;

                    // for each source point in the X axis
                    #ifdef GPU_OPENACC
                    #pragma acc loop seq
                    #endif
                    for(size_t j = src_x_begin; j <= src_x_end; j++){

                        f_type kws = src_points_values[kws_index_z] * src_points_values[kws_index_x];

                        // current source point in the grid
                        size_t domain_offset = i * nx + j;
                        size_t next_snapshot = next_t * domain_size + domain_offset;
                
                        // parameter to be used
                        f_type slowness = 1.0 / (velocity[domain_offset] * velocity[domain_offset]);

                        // denominator with damp coefficient
                        f_type denominator = (1.0 + damp[domain_offset] * dt / 2);

                        f_type value = dtSquared / slowness * kws * wavelet[wavelet_offset] / denominator;

                        #if defined(CPU_OPENMP) || defined(GPU_OPENMP)
                        #pragma omp atomic
                        #endif

                        #ifdef GPU_OPENACC
                        #pragma acc atomic update
                        #endif
                        v[next_snapshot] += value;

                        kws_index_x++;
                    }
                    kws_index_z++;
                }
            }
        }


        /*
            Section 3: add boundary conditions (z_before, z_after, x_before, x_after)
            0 - no boundary condition
            1 - null dirichlet
            2 - null neumann
        */
        size_t z_before = boundary_conditions[0];
        size_t z_after = boundary_conditions[1];
        size_t x_before = boundary_conditions[2];
        size_t x_after = boundary_conditions[3];

        // boundary conditions on the left and right (fixed in X)
        #ifdef CPU_OPENMP
        #pragma omp parallel for simd
        #endif

        #ifdef GPU_OPENMP
        #pragma omp target teams distribute parallel for
        #endif

        #ifdef GPU_OPENACC
        #pragma acc parallel loop present(v)
        #endif

        for(size_t i = stencil_radius; i < nz - stencil_radius; i++){

            // null dirichlet on the left
            if(x_before == 1){
                size_t domain_offset = i * nx + stencil_radius;
                size_t next_snapshot = next_t * domain_size + domain_offset;
                v[next_snapshot] = 0.0;
            }

            // null neumann on the left
            if(x_before == 2){
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t ir = 1; ir <= stencil_radius; ir++){
                    size_t domain_offset = i * nx + stencil_radius;
                    size_t next_snapshot = next_t * domain_size + domain_offset;
                    v[next_snapshot - ir] = v[next_snapshot + ir];
                }
            }

            // null dirichlet on the right
            if(x_after == 1){
                size_t domain_offset = i * nx + (nx - stencil_radius - 1);
                size_t next_snapshot = next_t * domain_size + domain_offset;
                v[next_snapshot] = 0.0;
            }

            // null neumann on the right
            if(x_after == 2){
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t ir = 1; ir <= stencil_radius; ir++){
                    size_t domain_offset = i * nx + (nx - stencil_radius - 1);
                    size_t next_snapshot = next_t * domain_size + domain_offset;
                    v[next_snapshot + ir] = v[next_snapshot - ir];
                }
            }

        }

        // boundary conditions on the top and bottom (fixed in Z)
        #ifdef CPU_OPENMP
        #pragma omp parallel for simd
        #endif

        #ifdef GPU_OPENMP
        #pragma omp target teams distribute parallel for
        #endif

        #ifdef GPU_OPENACC
        #pragma acc parallel loop present(v)
        #endif

        for(size_t j = stencil_radius; j < nx - stencil_radius; j++){

            // null dirichlet on the top
            if(z_before == 1){
                size_t domain_offset = stencil_radius * nx + j;
                size_t next_snapshot = next_t * domain_size + domain_offset;
                v[next_snapshot] = 0.0;
            }

            // null neumann on the top
            if(z_before == 2){
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t ir = 1; ir <= stencil_radius; ir++){
                    size_t domain_offset = stencil_radius * nx + j;
                    size_t next_snapshot = next_t * domain_size + domain_offset;
                    v[next_snapshot - (ir * nx)] = v[next_snapshot + (ir * nx)];
                }
            }

            // null dirichlet on the bottom
            if(z_after == 1){
                size_t domain_offset = (nz - stencil_radius - 1) * nx + j;
                size_t next_snapshot = next_t * domain_size + domain_offset;
                v[next_snapshot] = 0.0;
            }

            // null neumann on the bottom
            if(z_after == 2){
                #ifdef GPU_OPENACC
                #pragma acc loop seq
                #endif
                for(size_t ir = 1; ir <= stencil_radius; ir++){
                    size_t domain_offset = (nz - stencil_radius - 1) * nx + j;
                    size_t next_snapshot = next_t * domain_size + domain_offset;
                    v[next_snapshot + (ir * nx)] = v[next_snapshot - (ir * nx)];
                }
            }

        }

        /*
            Section 4: gradient calculation
        */

        #ifdef CPU_OPENMP
        #pragma omp parallel for simd
        #endif

        #ifdef GPU_OPENMP
        #pragma omp target teams distribute parallel for collapse(2)
        #endif

        #ifdef GPU_OPENACC
        #pragma acc parallel loop collapse(2) present(v,u,grad)
        #endif

        for(size_t i = stencil_radius; i < nz - stencil_radius; i++) {
            for(size_t j = stencil_radius; j < nx - stencil_radius; j++) {
                // index of the current point in the grid
                size_t domain_offset = i * nx + j;

                size_t prev_snapshot = prev_t * domain_size + domain_offset;
                size_t current_snapshot = current_t * domain_size + domain_offset;
                size_t next_snapshot = next_t * domain_size + domain_offset;

                f_type v_second_time_derivative = (v[prev_snapshot] - 2.0 * v[current_snapshot] + v[next_snapshot]) / dtSquared;

                // update gradient
                size_t current_point_u = (n-1) * domain_size + domain_offset;
                grad[domain_offset] -= v_second_time_derivative * u[current_point_u];
            }
        }

    }
    

    #ifdef GPU_OPENMP    
    #pragma omp target exit data map(from: grad[:domain_size])
    #pragma omp target exit data map(delete: u[:u_size])
    #pragma omp target exit data map(delete: v[:v_size])
    #pragma omp target exit data map(delete: velocity[:domain_size])
    #pragma omp target exit data map(delete: damp[:domain_size])
    #pragma omp target exit data map(delete: coeff[:stencil_radius+1])
    #pragma omp target exit data map(delete: src_points_interval[:src_points_interval_size])
    #pragma omp target exit data map(delete: src_points_values[:src_points_values_size])
    #pragma omp target exit data map(delete: src_points_values_offset[:num_sources])    
    #pragma omp target exit data map(delete: wavelet[:wavelet_size * wavelet_count])
    #endif

    #ifdef GPU_OPENACC    
    #pragma acc exit data copyout(grad[:domain_size])    
    #pragma acc exit data delete(grad[:domain_size])
    #pragma acc exit data delete(u[:u_size])
    #pragma acc exit data delete(v[:v_size])
    #pragma acc exit data delete(velocity[:domain_size])
    #pragma acc exit data delete(damp[:domain_size])
    #pragma acc exit data delete(coeff[:stencil_radius+1])
    #pragma acc exit data delete(src_points_interval[:src_points_interval_size])
    #pragma acc exit data delete(src_points_values[:src_points_values_size])
    #pragma acc exit data delete(src_points_values_offset[:num_sources])    
    #pragma acc exit data delete(wavelet[:wavelet_size * wavelet_count])
    #endif

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    return exec_time;
}
