#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define THREADS_NUM_Y TY
#define THREADS_NUM_X TX

#define checkErrorCuda(ans) { gpuCheck((ans), __FILE__, __LINE__); }

inline void gpuCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU Check: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};

// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
   typedef float f_type;
#elif defined(DOUBLE)
   typedef double f_type;
#endif


__global__ void kernel_ExchangeValue(size_t nx, size_t nz,
                       size_t current_t, size_t next_t,
                       f_type *u){
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
         
    if( (i >= nz ) || (j >= nx ) )
        return;

    size_t domain_size = nz * nx;

    // exchange of values ​​required
    
    // index of the current point in the grid
    size_t domain_offset = i * nx + j;

    size_t current_snapshot = current_t * domain_size + domain_offset;
    size_t next_snapshot = next_t * domain_size + domain_offset;

    f_type aux = u[current_snapshot];
    u[current_snapshot] = u[next_snapshot];
    u[next_snapshot] = aux;                    
}


__global__
void kernel_BoundaryCond_LeftRight(size_t nx, size_t nz,
                                   size_t z_before, size_t z_after,
                                   size_t x_before, size_t x_after,
                                   size_t next_t, size_t stencil_radius, 
                                   f_type *u){

    int i = blockIdx.y * blockDim.y + threadIdx.y + stencil_radius;
         
    if( i >= (nz - stencil_radius) )
        return;

    size_t domain_size = nz * nx;

    // null dirichlet on the left
    if(x_before == 1){
        size_t domain_offset = i * nx + stencil_radius;
        size_t next_snapshot = next_t * domain_size + domain_offset;
        u[next_snapshot] = 0.0;
    }

    // null neumann on the left
    if(x_before == 2){
        for(size_t ir = 1; ir <= stencil_radius; ir++){
            size_t domain_offset = i * nx + stencil_radius;
            size_t next_snapshot = next_t * domain_size + domain_offset;
            u[next_snapshot - ir] = u[next_snapshot + ir];
        }
    }

    // null dirichlet on the right
    if(x_after == 1){
        size_t domain_offset = i * nx + (nx - stencil_radius - 1);
        size_t next_snapshot = next_t * domain_size + domain_offset;
        u[next_snapshot] = 0.0;
    }

    // null neumann on the right
    if(x_after == 2){
        for(size_t ir = 1; ir <= stencil_radius; ir++){
            size_t domain_offset = i * nx + (nx - stencil_radius - 1);
            size_t next_snapshot = next_t * domain_size + domain_offset;
            u[next_snapshot + ir] = u[next_snapshot - ir];
        }
    }
    
}

__global__
void kernel_BoundaryCond_TopBottom(size_t nx, size_t nz,
                                  size_t z_before, size_t z_after,
                                  size_t x_before, size_t x_after,
                                  size_t next_t, size_t stencil_radius, 
                                  f_type *u){
                                                
    int j = blockIdx.x * blockDim.x + threadIdx.x + stencil_radius;
         
    if( j  >= (nx - stencil_radius) )
        return;

    size_t domain_size = nz * nx;

    // null dirichlet on the top
    if(z_before == 1){
        size_t domain_offset = stencil_radius * nx + j;
        size_t next_snapshot = next_t * domain_size + domain_offset;
        u[next_snapshot] = 0.0;
    }

    // null neumann on the top
    if(z_before == 2){
        for(size_t ir = 1; ir <= stencil_radius; ir++){
            size_t domain_offset = stencil_radius * nx + j;
            size_t next_snapshot = next_t * domain_size + domain_offset;
            u[next_snapshot - (ir * nx)] = u[next_snapshot + (ir * nx)];
        }
    }

    // null dirichlet on the bottom
    if(z_after == 1){
        size_t domain_offset = (nz - stencil_radius - 1) * nx + j;
        size_t next_snapshot = next_t * domain_size + domain_offset;
        u[next_snapshot] = 0.0;
    }

    // null neumann on the bottom
    if(z_after == 2){
        for(size_t ir = 1; ir <= stencil_radius; ir++){
            size_t domain_offset = (nz - stencil_radius - 1) * nx + j;
            size_t next_snapshot = next_t * domain_size + domain_offset;
            u[next_snapshot + (ir * nx)] = u[next_snapshot - (ir * nx)];
        }
    }
    
}


__global__ 
void kernel_ComputeReceive(size_t num_receivers, size_t *rec_points_interval,
                           size_t *rec_points_values_offset, f_type *rec_points_values,
                           size_t current_t, size_t nx, size_t nz, 
                           size_t n, f_type *receivers,f_type *u){

    int rec = blockIdx.x * blockDim.x + threadIdx.x;                             
    
    if(rec >= num_receivers)
        return;

    f_type sum = 0.0;
    size_t domain_size = nz * nx;
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
    //size_t rec_x_num_points = rec_x_end - rec_x_begin + 1;

    // pointer to rec value offset
    size_t offset_rec_kws_index_z = rec_points_values_offset[rec];

    // index of the Kaiser windowed sinc value of the receiver point
    size_t kws_index_z = offset_rec_kws_index_z;

    // for each receiver point in the Z axis
    for(size_t i = rec_z_begin; i <= rec_z_end; i++){
        size_t kws_index_x = offset_rec_kws_index_z + rec_z_num_points;

        // for each receiver point in the X axis
        for(size_t j = rec_x_begin; j <= rec_x_end; j++){

            f_type kws = rec_points_values[kws_index_z] * rec_points_values[kws_index_x];

            // current receiver point in the grid
            size_t domain_offset = i * nx + j;
            size_t current_snapshot = current_t * domain_size + domain_offset;
            sum += u[current_snapshot] * kws;

            kws_index_x++;
        }
        kws_index_z++;
    }

    size_t current_rec_n = (n-1) * num_receivers + rec;
    receivers[current_rec_n] = sum;
}

__global__ 
void kernel_AddSourceTerm(size_t n, size_t *src_points_interval, size_t src_points_interval_size,
                          f_type *wavelet, size_t wavelet_size, size_t wavelet_count,
                          size_t num_sources, f_type *src_points_values, 
                          size_t *src_points_values_offset, 
                          f_type *u, f_type *velocity, f_type dtSquared, 
                          size_t nx, size_t nz, size_t next_t){

    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if(src >= num_sources)
        return;

    size_t domain_size = nz * nx;
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
        for(size_t i = src_z_begin; i <= src_z_end; i++){
            size_t kws_index_x = offset_src_kws_index_z + src_z_num_points;

            // for each source point in the X axis
            for(size_t j = src_x_begin; j <= src_x_end; j++){

                f_type kws = src_points_values[kws_index_z] * src_points_values[kws_index_x];

                // current source point in the grid
                size_t domain_offset = i * nx + j;
                size_t next_snapshot = next_t * domain_size + domain_offset;

                f_type value = dtSquared * velocity[domain_offset] * velocity[domain_offset] * kws * wavelet[wavelet_offset];

                // u[next_snapshot] += value;
                atomicAdd(&u[next_snapshot], value);

                kws_index_x++;
            }
            kws_index_z++;
        }
    }
}


__global__ 
void kernel_UpdateWavefield(size_t prev_t, size_t current_t, size_t next_t,
                            size_t stencil_radius, size_t nz, size_t nx, f_type dt,
                            f_type dzSquared, f_type dxSquared, f_type dtSquared,
                            f_type *u, f_type *velocity, f_type *coeff, f_type *damp){
    // nz --> vertical
    // nx --> horizontal 
    int i = blockIdx.y * blockDim.y + threadIdx.y + stencil_radius;
    int j = blockIdx.x * blockDim.x + threadIdx.x + stencil_radius;
         
    if( (i >= (nz - stencil_radius) )  || (j  >= (nx - stencil_radius) ) )
        return;

    size_t domain_size = nz * nx;
    size_t domain_offset = i * nx + j;

    //==================Passar para CPU===============
    size_t prev_snapshot = prev_t * domain_size + domain_offset;
    size_t current_snapshot = current_t * domain_size + domain_offset;
    size_t next_snapshot = next_t * domain_size + domain_offset;    

    // stencil code to update grid
    f_type value = 0.0;

    f_type sum_x = coeff[0] * u[current_snapshot];
    f_type sum_z = coeff[0] * u[current_snapshot];

    // radius of the stencil
    #pragma unroll 4
    for(int ir = 1; ir <= stencil_radius; ir++){
        //neighbors in the horizontal direction
        sum_x += coeff[ir] * (u[current_snapshot + ir] + u[current_snapshot - ir]);

        //neighbors in the vertical direction
        sum_z += coeff[ir] * (u[current_snapshot + (ir * nx)] + u[current_snapshot - (ir * nx)]);
    }

    value += sum_x/dxSquared + sum_z/dzSquared;

    //denominator with damp coefficient
    f_type denominator = (1.0 + damp[domain_offset] * dt);

    value *= (dtSquared * velocity[domain_offset] * velocity[domain_offset]) / denominator;

    u[next_snapshot] = 2.0 / denominator * u[current_snapshot] - ((1.0 - damp[domain_offset] * dt) / denominator) * u[prev_snapshot] + value;
}

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
    // nz --> vertical
    // nx --> horizontal 
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

    //Device variables
    f_type *d_u;
    f_type *d_velocity;
    f_type *d_damp;
    f_type *d_coeff;
    size_t *d_src_points_interval;
    f_type *d_src_points_values;
    size_t *d_src_points_values_offset;
    size_t *d_rec_points_interval;
    f_type *d_rec_points_values;
    size_t *d_rec_points_values_offset;
    f_type *d_wavelet;
    f_type *d_receivers;

    // 1- Device memory allocation  
    // 2- Data transfer from CPU to GPU memory

    long nbytes;
    size_t u_size = num_snapshots * domain_size;
    size_t shot_record_size = wavelet_size * num_receivers;

    nbytes = sizeof(f_type) * u_size;
    checkErrorCuda( cudaMalloc(&d_u, nbytes) );
    checkErrorCuda( cudaMemcpy(d_u, u, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * domain_size;
    checkErrorCuda( cudaMalloc( &d_velocity, nbytes) );
    checkErrorCuda( cudaMemcpy(d_velocity, velocity, nbytes, cudaMemcpyHostToDevice) );

    checkErrorCuda( cudaMalloc(&d_damp, nbytes) );
    checkErrorCuda( cudaMemcpy(d_damp, damp, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * (stencil_radius+1);
    checkErrorCuda( cudaMalloc( &d_coeff, nbytes) );
    checkErrorCuda( cudaMemcpy(d_coeff, coeff, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(size_t) * src_points_interval_size;
    checkErrorCuda( cudaMalloc(&d_src_points_interval, nbytes) );
    checkErrorCuda( cudaMemcpy(d_src_points_interval, src_points_interval, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * src_points_values_size;
    checkErrorCuda( cudaMalloc(&d_src_points_values, nbytes) );
    checkErrorCuda( cudaMemcpy(d_src_points_values, src_points_values, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(size_t) * num_sources;
    checkErrorCuda( cudaMalloc(&d_src_points_values_offset, nbytes) );
    checkErrorCuda( cudaMemcpy(d_src_points_values_offset, src_points_values_offset, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(size_t) * rec_points_interval_size;
    checkErrorCuda( cudaMalloc(&d_rec_points_interval, nbytes) );
    checkErrorCuda( cudaMemcpy(d_rec_points_interval, rec_points_interval, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * rec_points_values_size;
    checkErrorCuda( cudaMalloc( &d_rec_points_values, nbytes) );
    checkErrorCuda( cudaMemcpy(d_rec_points_values, rec_points_values, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(size_t) * num_receivers;
    checkErrorCuda( cudaMalloc(&d_rec_points_values_offset, nbytes) );
    checkErrorCuda( cudaMemcpy(d_rec_points_values_offset, rec_points_values_offset, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * wavelet_size * wavelet_count;
    checkErrorCuda( cudaMalloc( &d_wavelet, nbytes) );
    checkErrorCuda( cudaMemcpy(d_wavelet, wavelet, nbytes, cudaMemcpyHostToDevice) );

    nbytes = sizeof(f_type) * shot_record_size;
    checkErrorCuda( cudaMalloc( &d_receivers, nbytes) );
    checkErrorCuda( cudaMemcpy(d_receivers, receivers, nbytes, cudaMemcpyHostToDevice) );

    dim3 block = dim3(THREADS_NUM_X, THREADS_NUM_Y);
    dim3 grid;

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

        /*
            Section 1: update the wavefield according to the acoustic wave equation
        */       
        grid  = dim3( ceilf( nx / (float)block.x ) ,  ceilf( nz / (float) block.y) );

        kernel_UpdateWavefield<<<grid, block>>>(prev_t, current_t, next_t,
                                                stencil_radius, nz, nx, dt,
                                                dzSquared, dxSquared, dtSquared,
                                                d_u, d_velocity, d_coeff, d_damp );   


        #if defined(DEBUG)
            checkErrorCuda( cudaDeviceSynchronize() );
        #endif
       
        /*
            Section 2: add the source term
            NUM_SOURCE --> linear
        */
        grid  = dim3( ceilf( num_sources / (float) block.x ));

        kernel_AddSourceTerm<<<grid, block.x>>>(n, d_src_points_interval, src_points_interval_size,
                                              d_wavelet, wavelet_size, wavelet_count, num_sources, 
                                              d_src_points_values, d_src_points_values_offset, 
                                              d_u, d_velocity, dtSquared, nx, nz, next_t);    

        #if defined(DEBUG)
            checkErrorCuda( cudaDeviceSynchronize() );
        #endif

        /*
            Section 3: add boundary conditions (z_before, z_after, x_before, x_after)
            0 - no boundary condition
            1 - null dirichlet
            2 - null neumann
        */
        size_t z_before = boundary_conditions[0];
        size_t z_after  = boundary_conditions[1];
        size_t x_before = boundary_conditions[2];
        size_t x_after  = boundary_conditions[3];


       grid  = dim3(1, ceilf( nz / (float) block.y ) );
       
       kernel_BoundaryCond_LeftRight<<<grid, dim3(1, block.y)>>>(nx, nz,
                                                                 z_before, z_after,
                                                                 x_before, x_after,
                                                                 next_t, stencil_radius, 
                                                                 d_u);

        #if defined(DEBUG)
            checkErrorCuda( cudaDeviceSynchronize() );
        #endif
       
        grid  = dim3(ceilf( nx / (float) block.x ) );
       
        kernel_BoundaryCond_TopBottom<<<grid, block.x>>>(nx, nz,
                                                        z_before, z_after,
                                                        x_before, x_after,
                                                        next_t, stencil_radius, d_u);

        #if defined(DEBUG)
            checkErrorCuda( cudaDeviceSynchronize() );
        #endif       

        /*
            Section 4: compute the receivers
        */
       grid  = dim3( ceilf( num_receivers / (float) block.x ));

       kernel_ComputeReceive<<<grid, block.x>>>(num_receivers, d_rec_points_interval,
                                                d_rec_points_values_offset, d_rec_points_values,
                                                current_t, nx, nz, n,
                                                d_receivers, d_u);
        
        #if defined(DEBUG)
            checkErrorCuda( cudaDeviceSynchronize() );
        #endif

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

                    kernel_ExchangeValue<<<grid, block>>>(nx, nz, current_t, next_t, d_u);

                    #if defined(DEBUG)
                        checkErrorCuda( cudaDeviceSynchronize() );
                    #endif
                }

            }else{
                prev_t = current_t;
                current_t = next_t;
                next_t = prev_t;
            }
        }
    }

    // Data transfer from GPU to CPU memory
    nbytes = sizeof(f_type) * u_size;
    checkErrorCuda( cudaMemcpy(u, d_u, nbytes, cudaMemcpyDeviceToHost) );

    nbytes = sizeof(f_type) * shot_record_size;
    checkErrorCuda( cudaMemcpy(receivers, d_receivers, nbytes, cudaMemcpyDeviceToHost) );
  
    //Free device memory 
    checkErrorCuda( cudaFree(d_u) );
    checkErrorCuda( cudaFree(d_velocity) );
    checkErrorCuda( cudaFree(d_damp) );
    checkErrorCuda( cudaFree(d_coeff) );
    checkErrorCuda( cudaFree(d_src_points_interval) );
    checkErrorCuda( cudaFree(d_src_points_values) );
    checkErrorCuda( cudaFree(d_src_points_values_offset) );
    checkErrorCuda( cudaFree(d_rec_points_interval) );
    checkErrorCuda( cudaFree(d_rec_points_values) );
    checkErrorCuda( cudaFree(d_rec_points_values_offset) );
    checkErrorCuda( cudaFree(d_wavelet) );
    checkErrorCuda( cudaFree(d_receivers) );

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

 

    return exec_time;
}
