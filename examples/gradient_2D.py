from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np

# domain size (meters) in each axis
domain_size = 1000

# grid spacing in each axis
spacing = 10

# space order
space_order = 4

# dtype
dtype = np.float32

# propagation time
propagation_time = 0.5

# time step variation
dt = 0.001

# compiler options
compiler_options = {
    'c': {
        'cc': 'gcc',
        'language': 'c',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    },
    'cpu_openmp': {
        'cc': 'gcc',
        'language': 'cpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared -fopenmp'
    },
    'gpu_openmp': {
        'cc': 'clang',
        'language': 'gpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -fopenmp \
                   -fopenmp-targets=nvptx64-nvidia-cuda \
                   -Xopenmp-target -march=sm_75'
    },
    'gpu_openacc': {
        'cc': 'pgcc',
        'language': 'gpu_openacc',
        'cflags': '-O3 -fPIC -acc:gpu -gpu=pinned -mp'
    },
}

selected_compiler = compiler_options['c']

def create_solver(saving_stride, velocity_model):
    
    # set compiler options
    compiler = Compiler(
        cc=selected_compiler['cc'],
        language=selected_compiler['language'],
        cflags=selected_compiler['cflags']
    )
    
    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, domain_size, 0, domain_size),
        grid_spacing=(spacing, spacing),
        velocity_model=velocity_model,
        space_order=space_order,
        dtype=dtype
    )

    # damping extension (meters)
    damping = domain_size // 2
    
    # config boundary conditions
    # (none,  null_dirichlet or null_neumann)   
    space_model.config_boundary(
        #damping_length=(damping, damping, damping, damping),
        damping_length=10,
        boundary_condition=(
            "null_neumann", "null_dirichlet",
            "null_dirichlet", "null_dirichlet"
        ),
        damping_polynomial_degree=3,
        damping_alpha=0.002
    )

    # create the time model
    time_model = TimeModel(
        space_model=space_model,
        tf=propagation_time,
        dt=dt,
        saving_stride=saving_stride
    )
    
    print("DT", time_model.dt)

    # create the set of sources
    source = Source(
        space_model,
        coordinates=[(20, domain_size//2)],
        window_radius=4
    )

    # crete the set of receivers
    receiver = Receiver(
        space_model=space_model,
        #coordinates=[(domain_size-20, i) for i in range(0, domain_size, 10)],
        coordinates=[(domain_size-20, i) for i in np.linspace(0, domain_size, num=101)],
        window_radius=4
    )

    # create a ricker wavelet with 10hz of peak frequency
    ricker = RickerWavelet(10.0, time_model)

    # create the solver
    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=ricker,
        compiler=compiler
    )
    
    return solver

def calculate_true_seimogram(velocity_model):
    
    # true model solver
    solver = create_solver(saving_stride=0, velocity_model=velocity_model)
    
    u_true, recv_true = solver.forward()

    print("Shape u", u_true.shape)  
    
    plot_velocity_model(solver.space_model.velocity_model,
                        sources=solver.sources.grid_positions,
                        receivers=solver.receivers.grid_positions,
                        file_name="true_velocity_model")
    
    plot_wavefield(u_true[-1], file_name="true_final_wavefield")
    plot_shotrecord(recv_true, file_name="true_seismogram")
    
    return recv_true

def compute_gradient(velocity_model, recv_true):
    
    # true model solver
    solver = create_solver(saving_stride=1, velocity_model=velocity_model)
    
    # run the forward computation
    u_full, recv_sim = solver.forward()
    
    plot_velocity_model(solver.space_model.velocity_model,                  
                        file_name="smooth_velocity_model")
    
    
    plot_wavefield(u_full[-1], file_name="smooth_final_wavefield")
    plot_shotrecord(recv_sim, file_name="smooth_seismogram")
    
    # residual
    residual = recv_sim - recv_true
    
    plot_shotrecord(residual, file_name="residual_seismogram")
    
    # compute the gradient
    grad = solver.gradient(u_full, residual)
    
    plot_velocity_model(grad, file_name="gradient")
    
    return grad 

def camembert_velocity_model(grid_size, radius):
    
    shape=(grid_size, grid_size)
    
    vel = np.zeros(shape, dtype=dtype)
    vel[:] = 2500

    a, b = shape[0] / 2, shape[1] / 2
    z, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
    vel[z*z + x*x <= radius*radius] = 3000
    
    return vel
    

if __name__ == "__main__":
    
    grid_size = domain_size // spacing + 1

    # True velocity model
    # Camembert model    
    tru_vel = camembert_velocity_model(grid_size, radius=15)
   
    # Smooth velocity model
    smooth_vel = np.zeros(shape=(grid_size, grid_size), dtype=dtype)
    smooth_vel[:] = 2000
    
    # calculate the true (observed) seismogram
    recv_true = calculate_true_seimogram(velocity_model=tru_vel)
    
    # compute the adjoint and gradient
    grad = compute_gradient(velocity_model=smooth_vel, recv_true=recv_true)



    #print("Timesteps:", time_model.timesteps)

    # run the forward
    #u_full, recv = solver.forward()

    #print("u_full shape:", u_full.shape)
    #plot_velocity_model(space_model.velocity_model)
    #plot_wavefield(u_full[-1])
    #plot_shotrecord(recv)

    #grad = solver.gradient(u_full, recv)

    #plot_velocity_model(grad, file_name="grad")

    #print(np.mean(grad))
