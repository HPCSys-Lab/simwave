#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <occa.hpp>

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif

// forward_2D_constant_density
extern "C" double forward(f_type *u, f_type *velocity, f_type *damp,
               f_type *wavelet, size_t wavelet_size, size_t wavelet_count,
               f_type *coeff, size_t *boundary_conditions,
               size_t *src_points_interval, size_t src_points_interval_size,
               f_type *src_points_values, size_t src_points_values_size,
               size_t *src_points_values_offset,
               size_t *rec_points_interval, size_t rec_points_interval_size,
               f_type *rec_points_values, size_t rec_points_values_size,
               size_t *rec_points_values_offset,
               f_type *receivers, size_t num_sources, size_t num_receivers,
               size_t nz, size_t nx, f_type dz, f_type dx,
               size_t saving_stride, f_type dt,
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

    occa::device device;

    // device is CPU and uses OCCA OpenMP mode
    #ifdef CPU_OCCA
    
    device.setup({
        {"mode", "OpenMP"}        
    });

    #endif

    // device is GPU and uses OCCA Cuda mode
    #ifdef GPU_OCCA

    #ifndef DEVICEID
    #define DEVICEID 0
    #endif
    
    device.setup({
        {"mode"     , "CUDA"},
        {"device_id", DEVICEID},
    }); 

    #endif

    // register f_type in occa
    //occa::dtype_t d_f_type("f_type", sizeof(f_type));
    //d_f_type.registerType();

    size_t shot_record_size = wavelet_size * num_receivers;
    size_t u_size = num_snapshots * domain_size;    

    // allocate memory on device
    occa::memory d_u = device.malloc<f_type>(u_size);
    occa::memory d_velocity = device.malloc<f_type>(domain_size);
    occa::memory d_damp = device.malloc<f_type>(domain_size);
    occa::memory d_coeff = device.malloc<f_type>(stencil_radius+1);
    occa::memory d_src_points_interval = device.malloc<long>(src_points_interval_size);
    occa::memory d_src_points_values = device.malloc<f_type>(src_points_values_size);
    occa::memory d_src_points_values_offset = device.malloc<long>(num_sources);
    occa::memory d_rec_points_interval = device.malloc<long>(rec_points_interval_size);
    occa::memory d_rec_points_values = device.malloc<f_type>(rec_points_values_size);
    occa::memory d_rec_points_values_offset = device.malloc<long>(num_receivers);
    occa::memory d_wavelet = device.malloc<f_type>(wavelet_size * wavelet_count);
    occa::memory d_receivers = device.malloc<f_type>(shot_record_size);

    // copy memory to the device
    d_u.copyFrom(u);
    d_velocity.copyFrom(velocity);
    d_damp.copyFrom(damp);
    d_coeff.copyFrom(coeff);
    d_src_points_interval.copyFrom(src_points_interval);
    d_src_points_values.copyFrom(src_points_values);
    d_src_points_values_offset.copyFrom(src_points_values_offset);
    d_rec_points_interval.copyFrom(rec_points_interval);
    d_rec_points_values.copyFrom(rec_points_values);
    d_rec_points_values_offset.copyFrom(rec_points_values_offset);
    d_wavelet.copyFrom(wavelet);
    d_receivers.copyFrom(receivers);

    // Compile the kernels at run-time    
    occa::kernel stencil = device.buildKernel("stencil.okl", "stencil");
    occa::kernel source_injection = device.buildKernel("source_injection.okl", "source_injection");
    occa::kernel boundary_conditions_1 = device.buildKernel("boundary_conditions_1.okl", "boundary_conditions_1");
    occa::kernel boundary_conditions_2 = device.buildKernel("boundary_conditions_2.okl", "boundary_conditions_2");
    occa::kernel sismogram = device.buildKernel("sismogram.okl", "sismogram");
    occa::kernel swap_grid_in_even_stride = device.buildKernel("swap_grid_in_even_stride.okl", "swap_grid_in_even_stride");

    // wavefield modeling
    for(size_t n = begin_timestep; n <= end_timestep; n++) {

        // no saving case
        if(saving_stride == 0){
            prev_t = (n - 1) % 3;
            current_t = n % 3;
            next_t = (n + 1) % 3;
        }else{
            // all timesteps saving case
            if(saving_stride == 1){
                prev_t = n - 1;
                current_t = n;
                next_t = n + 1;
            }
        }

        // device kernel for fot section 1: update the wavefield according to the acoustic wave equation        
        stencil(
            stencil_radius, nz, nx, domain_size, prev_t, current_t, next_t,
            dzSquared, dxSquared, dtSquared, dt,
            d_u, d_velocity, d_coeff, d_damp
        );
        
        // device kernel for section 2: add the source term
        source_injection(
            num_sources, n, wavelet_count, next_t, nx, domain_size, dtSquared,
            d_wavelet, d_src_points_interval.cast(occa::dtype::long_), d_src_points_values_offset.cast(occa::dtype::long_), d_src_points_values, d_velocity, d_u
        );

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

        // device kernel for section 3.1
        boundary_conditions_1(stencil_radius, nz, nx, x_before, x_after, next_t, domain_size, d_u);

        // device kernel for section 3.2
        boundary_conditions_2(stencil_radius, nz, nx, z_before, z_after, next_t, domain_size, d_u);

        // device kernel for section 4: compute the receivers
        sismogram(
            num_receivers, n, current_t, nx, domain_size, 
            d_rec_points_interval.cast(occa::dtype::long_), d_rec_points_values_offset.cast(occa::dtype::long_), d_rec_points_values,
            d_u, d_receivers
        );        

        // stride timesteps saving case
        if(saving_stride > 1){
            // shift the pointer
            if(n % saving_stride == 1){

                prev_t = current_t;
                current_t += 1;
                next_t += 1;

                // even stride adjust case
                if(saving_stride % 2 == 0 && n < end_timestep){
                    size_t swap = current_t;
                    current_t = next_t;
                    next_t = swap;

                    // device kernel for grid swapping
                    swap_grid_in_even_stride(nz, nx, domain_size, current_t, next_t, d_u);
                    
                }

            }else{
                prev_t = current_t;
                current_t = next_t;
                next_t = prev_t;
            }
        }

    }

    // copy result to the host
    d_u.copyTo(u);
    d_receivers.copyTo(receivers);

    device.finish();   

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    return exec_time;
}
