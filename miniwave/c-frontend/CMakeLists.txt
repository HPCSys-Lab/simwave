cmake_minimum_required(VERSION 3.18)

project(miniwave)

# Set the kernel source file and header
configure_file(${CMAKE_SOURCE_DIR}/selected_kernel.h.in
               ${CMAKE_BINARY_DIR}/selected_kernel.h)
include_directories(${CMAKE_BINARY_DIR})

add_executable(miniwave ../${KERNEL_SOURCE} miniwave.c)

# Include miniwave header files (sequential.h)
target_include_directories(miniwave PUBLIC ${CMAKE_SOURCE_DIR}/../kernels)

# HDF5 commands
set(HDF5DIR "$ENV{HOME}/hdf5/build")
target_include_directories(miniwave PRIVATE ${HDF5DIR}/include)
target_link_directories(miniwave PRIVATE ${HDF5DIR}/lib)
target_link_libraries(miniwave PRIVATE libhdf5.so libhdf5_cpp.so)
