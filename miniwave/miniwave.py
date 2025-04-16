import numpy as np
import argparse
import sys
from utils import Model, Compiler, Kernel, plot
from utils.properties import Properties

def get_args(args=sys.argv[1:]):
    
    parser = argparse.ArgumentParser(description='How to use this program')
    
    parser.add_argument("--file", type=str, default='kernels/sequential.c',
                        help="Path to the Kernel file")
    
    parser.add_argument("--grid_size", type=int, default=256,
                        help="Grid size")
    
    parser.add_argument("--num_timesteps", type=int, default=400,
                        help="Number of timesteps")
    
    parser.add_argument("--language", type=str, default="c", choices=['c', 'openmp', 'openmp_cpu', 'openacc', 'cuda', 'python', 'mpi', 'mpi_cuda', 'ompc'],
                        help="Language: c, openmp, openacc, cuda, python, ompc, mpi, mpi_cuda")
    
    parser.add_argument("--space_order", type=int, default=2,
                        help="Space order")
    
    parser.add_argument("--block_size_1", type=int, default=1,
                        help="GPU block size in the first axis")
    
    parser.add_argument("--block_size_2", type=int, default=1,
                        help="GPU block size in the second axis")
    
    parser.add_argument("--block_size_3", type=int, default=1,
                        help="GPU block size in the third axis") 
    
    parser.add_argument("--sm", type=int, default=75,
                        help="Cuda capability") 
    
    parser.add_argument("--fast_math", default=False, action="store_true" , help="Enable --fast-math flag")
    
    parser.add_argument("--plot", default=False, action="store_true" , help="Enable ploting")
    
    parser.add_argument("--dtype", type=str, default="float64", help="Float Precision. float32 or float64 (default)")      
    
    
    parsed_args = parser.parse_args(args)

    return parsed_args


if __name__ == "__main__":
    
    args = get_args()
    
    # enable/disable fast math 
    fast_math = args.fast_math
    
    # cuda capability
    sm = args.sm 
    
    # language
    language = args.language
    
    # float precision
    dtype = args.dtype
    
    # create a compiler object
    compiler = Compiler(language=language, sm=sm, fast_math=fast_math)
    
    # define grid shape
    grid_size = (args.grid_size, args.grid_size, args.grid_size)
    
    vel_model = np.ones(shape=grid_size) * 1500.0
    
    model = Model(
        velocity_model=vel_model,
        grid_spacing=(10,10,10),
        dt=0.002,
        num_timesteps=args.num_timesteps,
        space_order=args.space_order,
        dtype=dtype
    )       
    
    # GPU block sizes  
    properties = Properties(
        block_size_1=args.block_size_1,
        block_size_2=args.block_size_2, 
        block_size_3=args.block_size_3
    )   
    
    solver = Kernel(
        file=args.file,        
        model=model,
        compiler=compiler,
        properties=properties
    )      
    
    # run the kernel
    exec_time, u = solver.run()
    
    # plot a slice
    if args.plot:
        slice = vel_model.shape[1] // 2        
        plot(u[:,slice,:])
    
    print(f"Execution time: {exec_time} seconds")
