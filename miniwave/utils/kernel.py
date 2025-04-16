import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import importlib.util
import sys
from typing import Tuple, Optional, Callable
from utils.model import Model
from utils.compiler import Compiler
from utils.properties import Properties
from utils.dataset_writer import DatasetWriter
import subprocess
import h5py


class Kernel:

    def __init__(
        self,
        file: str,
        model: Model,
        compiler: Optional[Compiler] = None,
        properties: Optional[Properties] = Properties(1, 1, 1),
    ):

        self._file = file
        self._model = model
        self._compiler = compiler
        self._properties = properties

        if self.language != "python" and compiler is None:
            raise Exception("Compiler can not be None")

    @property
    def file(self) -> str:
        return self._file

    @property
    def language(self) -> str:
        return self.compiler.language

    @property
    def model(self) -> Model:
        return self._model

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def properties(self) -> Properties:
        return self._properties

    def _import_python_lib(self) -> Callable:

        spec = importlib.util.spec_from_file_location(
            "kernel.forward",
            self.file
        )
        lib = importlib.util.module_from_spec(spec)
        sys.modules["kernel.forward"] = lib
        spec.loader.exec_module(lib)

        return lib.forward

    def _import_c_lib(self) -> Callable:

        dtype = "float32" if self.model.dtype == np.float32 else "float64"

        # add space order and dtype to properties
        self.properties.space_order = self.model.space_order
        self.properties.dtype = dtype

        shared_object = self.compiler.compile(self.file, self.properties)

        # load the library
        lib = ctypes.cdll.LoadLibrary(shared_object)

        if self.properties.dtype == "float32":

            argtypes = [
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # prev_u
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # next_u
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # vel_model
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # coeffs
                ctypes.c_float,  # d1
                ctypes.c_float,  # d2
                ctypes.c_float,  # d3
                ctypes.c_float,  # dt
                ctypes.c_int,  # n1
                ctypes.c_int,  # n2
                ctypes.c_int,  # n3
                ctypes.c_int,  # number of timesteps
                ctypes.c_int,  # radius of space order
                ctypes.c_int,  # block_size_1
                ctypes.c_int,  # block_size_2
                ctypes.c_int,  # block_size_3
            ]

        elif self.properties.dtype == "float64":

            argtypes = [
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # prev_u
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # next_u
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # vel_model
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # coeffs
                ctypes.c_double,  # d1
                ctypes.c_double,  # d2
                ctypes.c_double,  # d3
                ctypes.c_double,  # dt
                ctypes.c_int,  # n1
                ctypes.c_int,  # n2
                ctypes.c_int,  # n3
                ctypes.c_int,  # number of timesteps
                ctypes.c_int,  # radius of space order
                ctypes.c_int,  # block_size_1
                ctypes.c_int,  # block_size_2
                ctypes.c_int,  # block_size_3
            ]

        else:
            raise ValueError(
                "Unkown float precision. Must be float32 or float64"
            )

        forward = lib.forward
        forward.restype = ctypes.c_int
        forward.argtypes = argtypes

        return forward

    def _load_lib(self) -> Callable:

        if self.language == "python":
            return self._import_python_lib()
        else:
            return self._import_c_lib()

    def read_data_from_hdf5(self, filename):
        with h5py.File(filename, "r") as file:
            exec_time = file["/execution_time"][()]
            vector = file["/vector"][:]
        return exec_time, vector

    def print_setup(self):
        print("----------")
        print("Language: ", self.language)
        print("File: ", self.file)
        print(f"Dtype: {self.model.dtype} ({self.properties.dtype})")
        print("Grid size: ", self.model.grid_shape)
        print("Grid spacing: ", self.model.grid_spacing)
        print("Space order: ", self.model.space_order)
        print("-DSTENCIL_RADIUS: ", self.properties.stencil_radius)
        print("Iterations: ", self.model.num_timesteps)
        print("Block sizes: ", self.properties.block3)
        print("SM: ", self.compiler.sm)
        print("Fast Math: ", self.compiler.fast_math)
        print("----------")

    def run(self) -> Tuple[float, np.ndarray]:
        # return execution time and last wavefield

        # load the lib
        self._load_lib()

        # get args
        prev_u, next_u = self.model.u_arrays
        n1, n2, n3 = self.model.grid_shape
        d1, d2, d3 = self.model.grid_spacing
        stencil_radius = self.model.space_order // 2

        # print setup
        self.print_setup()

        # Generate HDF5 file

        data = {
            "prev_u": {"dataset_data": prev_u, "dataset_attributes": {}},
            "next_u": {"dataset_data": next_u, "dataset_attributes": {}},
            "vel_model": {
                "dataset_data": self.model.velocity_model,
                "dataset_attributes": {},
            },
            "coefficient": {
                "dataset_data": self.model.stencil_coefficients,
                "dataset_attributes": {},
            },
            "d1": {"dataset_data": d1, "dataset_attributes": {}},
            "d2": {"dataset_data": d2, "dataset_attributes": {}},
            "d3": {"dataset_data": d3, "dataset_attributes": {}},
            "dt": {"dataset_data": self.model.dt, "dataset_attributes": {}},
            "n1": {"dataset_data": n1, "dataset_attributes": {}},
            "n2": {"dataset_data": n2, "dataset_attributes": {}},
            "n3": {"dataset_data": n3, "dataset_attributes": {}},
            "iterations": {
                "dataset_data": self.model.num_timesteps,
                "dataset_attributes": {},
            },
            "stencil_radius": {
                "dataset_data": stencil_radius,
                "dataset_attributes": {},
            },
            "block_size_1": {
                "dataset_data": self.properties.block_size_1,
                "dataset_attributes": {},
            },
            "block_size_2": {
                "dataset_data": self.properties.block_size_2,
                "dataset_attributes": {},
            },
            "block_size_3": {
                "dataset_data": self.properties.block_size_3,
                "dataset_attributes": {},
            },
        }

        DatasetWriter.write_dataset(data, "c-frontend/data/miniwave_data.h5")

        KERNEL_SOURCE = self.file
        KERNEL_HEADER = self.file.split('.')[0] + ".h"
        CUDA_ARCH = self.compiler.sm
        KERNEL_TYPE = self.language

        # Compile C code
        subprocess.run("ls c-frontend", shell=True)
        subprocess.run(
            f"cmake -S c-frontend -B c-frontend/build/ \
            -DKERNEL_SOURCE={KERNEL_SOURCE} -DKERNEL_HEADER={KERNEL_HEADER} \
            -DKERNEL_TYPE={KERNEL_TYPE} -DCUDA_ARCHITECTURE={CUDA_ARCH}",
            shell=True
        )
        subprocess.run("cmake --build c-frontend/build/", shell=True)

        # run the forward function
        if self.language == "ompc":
            subprocess.run(
                "mpirun -np 4 offload-mpi-worker : \
                -np 1 ./c-frontend/build/miniwave",
                shell=True
            )
        elif self.language == "mpi" or self.language == "mpi_cuda":
            subprocess.run(
                "mpirun -np 4 ./c-frontend/build/miniwave",
                shell=True
            )
        else:
            subprocess.run("./c-frontend/build/miniwave", shell=True)

        exec_time, next_u = self.read_data_from_hdf5(
            "./c-frontend/data/results.h5"
        )

        return exec_time[0], next_u
