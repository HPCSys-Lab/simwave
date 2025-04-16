# Miniwave - Minimum Simwave

Python wrapper for forward wave propagation.

# Dependencies

`pip install numpy matplotlib findiff h5py`

```
mkdir /hdf5 && \
    cd /hdf5 && \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.3/src/CMake-hdf5-1.14.3.tar.gz && \
    tar xf CMake-hdf5-1.14.3.tar.gz && \
    cd CMake-hdf5-1.14.3 && \
    ./build-unix.sh && \
    yes | ./HDF5-1.14.3-Linux.sh && \
    cp -r HDF5-1.14.3-Linux/HDF_Group/HDF5/1.14.3/ ../build
```

# How to use

`python3 miniwave.py --help`

`python3 miniwave.py --file FILE --grid_size GRID_SIZE --num_timesteps NUM_TIMESTEPS --language {c,openmp,openacc,cuda,python} --space_order SPACE_ORDER`

