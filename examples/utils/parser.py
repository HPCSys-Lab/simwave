import argparse
import sys

def get_options(args=sys.argv[1:]):

    parser = argparse.ArgumentParser(description='How to use this program')
    parser.add_argument("-i", "--tool", type=str, default='sequential', choices=['sequential', 'cuda', 'openmp', 'openacc'], help="Implementation strategy. Options: sequential, cuda, openmp, openacc")
    parser.add_argument("-c", "--ccode", type=str, help="Path to the C code file")
    parser.add_argument("-vm", "--vmodel", type=str, help="Path to the velocity model file")
    parser.add_argument("-dm", "--dmodel", type=str, help="Path to the density model file")
    parser.add_argument("-d", "--grid", nargs="+", type=int, default=[512, 512], help="Number of grid points along each axis. Example: 512 512 512")
    parser.add_argument("-s", "--spacing", nargs="+", type=float, default=[20.0, 20.0], help="Spacing between points along each axis (in meters). Example: 20.0 20.0 20.0")
    parser.add_argument("-t", "--time", type=int, default=1000, help="Propagation time (in miliseconds)")
    parser.add_argument("-ps", "--print_steps", type=float, default=0, help="Interval of intermediate wavefield steps to save. If 0, It saves no intermediate wavefield")

    parsed_args = parser.parse_args(args)

    if len(parsed_args.grid) < 2 or len(parsed_args.grid) > 3:
        raise Exception("Grid dimension (--grid) must 2 (2D) or 3 (3D)")

    if len(parsed_args.spacing) != len(parsed_args.grid):
        raise Exception("Spacing (--spacing) must have the same number of axis as the grid (--grid)")

    return parsed_args
