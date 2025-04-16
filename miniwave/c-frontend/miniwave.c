#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hdf5.h"
#ifdef CONTAINS_MPI
    #include "mpi.h"
#endif
#include "selected_kernel.h"

hid_t open_hdf5_file(char *file) {
    return H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
}
hid_t open_hdf5_dataset(hid_t file_id, char *dataset) {
    return H5Dopen2(file_id, dataset, H5P_DEFAULT);
}
float *read_float_dataset(hid_t dataset_id) {
    hid_t dataspace;
    hsize_t dims_out[10];
    int rank, total_size;

    dataspace = H5Dget_space(dataset_id); /* dataspace handle */
    rank = H5Sget_simple_extent_ndims(dataspace);
    H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

    total_size = 1;
    for (size_t i = 0; i < rank; i++) {
        total_size *= dims_out[i];
    }

    float *dset_data = malloc(sizeof(float) * total_size);
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            dset_data);
    return dset_data;
}
float read_float_attribute(hid_t dataset_id, char *attribute_name) {
    // Gets attribute value from a dataset in a file
    // Ex: file example_data.h5 has a dataset named scalar_data
    //     which has an attribute named 'attribute_X'. To get 
    //     the value of this attribute:
    // 
    //     float attribute_X =
    //       read_float_attribute(scalar_data_dataset_id, "attribute_X");

    
    const char *attribute_value;
    // Get attribute value as string
    hid_t attribute_id = H5Aopen(dataset_id, attribute_name, H5P_DEFAULT);
    hid_t attribute_type = H5Aget_type(attribute_id);
    H5Aread(attribute_id, attribute_type, &attribute_value);
    // Convert attribute value to float
    return atof(attribute_value);
}
void close_hdf5_dataset(hid_t dataset_id) { H5Dclose(dataset_id); }
void close_hdf5_file(hid_t file_id) { H5Fclose(file_id); }

int write_hdf5_result(int n1, int n2, int n3, double execution_time, f_type* next_u) {
    hid_t h5t_type = H5T_NATIVE_FLOAT;
    #if defined(DOUBLE)
        h5t_type = H5T_NATIVE_DOUBLE;
    #endif

    // Create a new HDF5 file using the default properties
    hid_t file_id = H5Fcreate("c-frontend/data/results.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error creating file.\n");
        return 1;
    }

    // Define the dataspace for the vector dataset
    hsize_t vector_dims[3] = {n1,n2,n3};  // 3D vector
    hid_t vector_dataspace_id = H5Screate_simple(3, vector_dims, NULL);
    if (vector_dataspace_id < 0) {
        printf("Error creating vector dataspace.\n");
        H5Fclose(file_id);
        return 1;
    }

    // Create the vector dataset with default properties
    hid_t vector_dataset_id = H5Dcreate(file_id, "vector", h5t_type, vector_dataspace_id,
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (vector_dataset_id < 0) {
        printf("Error creating vector dataset.\n");
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    // Write the vector data to the dataset
    if (H5Dwrite(vector_dataset_id, h5t_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, next_u) < 0) {
        printf("Error writing vector data.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    // Define the dataspace for the execution time dataset
    hsize_t time_dims[1] = {1};  // Scalar
    hid_t time_dataspace_id = H5Screate_simple(1, time_dims, NULL);
    if (time_dataspace_id < 0) {
        printf("Error creating time dataspace.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    // Create the execution time dataset with default properties
    hid_t time_dataset_id = H5Dcreate(file_id, "execution_time", H5T_NATIVE_DOUBLE, time_dataspace_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (time_dataset_id < 0) {
        printf("Error creating time dataset.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Sclose(time_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    // Write the execution time to the dataset
    if (H5Dwrite(time_dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &execution_time) < 0) {
        printf("Error writing time data.\n");
        H5Dclose(vector_dataset_id);
        H5Sclose(vector_dataspace_id);
        H5Dclose(time_dataset_id);
        H5Sclose(time_dataspace_id);
        H5Fclose(file_id);
        return 1;
    }

    // Close the datasets, dataspaces, and file
    H5Dclose(vector_dataset_id);
    H5Sclose(vector_dataspace_id);
    H5Dclose(time_dataset_id);
    H5Sclose(time_dataspace_id);
    H5Fclose(file_id);
    return 0;
}

int main() {
    #ifdef CONTAINS_MPI
        MPI_Init(NULL, NULL);
    #endif

    int rank = 0;

    #ifdef CONTAINS_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    // Get the file id
    hid_t file_id = open_hdf5_file("c-frontend/data/miniwave_data.h5");
    // Read arguments
    hid_t vel_model_id = open_hdf5_dataset(file_id, "vel_model");
    hid_t next_u_id = open_hdf5_dataset(file_id, "next_u");
    hid_t prev_u_id = open_hdf5_dataset(file_id, "prev_u");
    hid_t coefficient_id = open_hdf5_dataset(file_id, "coefficient");
    hid_t scalar_data_id = open_hdf5_dataset(file_id, "scalar_data");
    float *vel_model = read_float_dataset(vel_model_id);
    float *next_u = read_float_dataset(next_u_id);
    float *prev_u = read_float_dataset(prev_u_id);
    float *coefficient = read_float_dataset(coefficient_id);
    float block_size_1 = read_float_attribute(scalar_data_id, "block_size_1");
    float block_size_2 = read_float_attribute(scalar_data_id, "block_size_2");
    float block_size_3 = read_float_attribute(scalar_data_id, "block_size_3");
    float d1 = read_float_attribute(scalar_data_id, "d1");
    float d2 = read_float_attribute(scalar_data_id, "d2");
    float d3 = read_float_attribute(scalar_data_id, "d3");
    float dt = read_float_attribute(scalar_data_id, "dt");
    float iterations = read_float_attribute(scalar_data_id, "iterations");
    float n1 = read_float_attribute(scalar_data_id, "n1");
    float n2 = read_float_attribute(scalar_data_id, "n2");
    float n3 = read_float_attribute(scalar_data_id, "n3");
    float stencil_radius = read_float_attribute(scalar_data_id, "stencil_radius");

    if (rank == 0) {
        printf("vel_model[0:50]:\n");
        for (size_t i = 0; i < 50; i++) {
            printf("%f ", vel_model[i]);
        }
        printf("\n");
        printf("next_u[0:50]:\n");
        for (size_t i = 0; i < 50; i++) {
            printf("%lf ", next_u[i]);
        }
        printf("\n");
        printf("prev_u[0:50]:\n");
        for (size_t i = 0; i < 50; i++) {
            printf("%lf ", prev_u[i]);
        }
        printf("\n");
        printf("coefficient:\n");
        for (size_t i = 0; i < 2; i++) {
            printf("%lf ", coefficient[i]);
        }
        printf("\n");
        printf("block_size_1: %f\n",block_size_1);
        printf("block_size_2: %f\n",block_size_2);
        printf("block_size_3: %f\n",block_size_3);
        printf("d1: %f\n",d1);
        printf("d2: %f\n",d2);
        printf("d3: %f\n",d3);
        printf("dt: %f\n",dt);
        printf("iterations: %f\n",iterations);
        printf("n1: %f\n",n1);
        printf("n2: %f\n",n2);
        printf("n3: %f\n",n3);
        printf("stencil_radius: %f\n",stencil_radius);
    }

    clock_t start_time = clock();

    forward(prev_u, next_u, vel_model, coefficient, d1, d2, d3, dt, n1, n2, n3, iterations, stencil_radius, block_size_1, block_size_2, block_size_3);

    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (rank == 0) {
        printf("\nprev_u[0:50]");
        for (size_t i = 0; i < 50; i++) {
            printf("%lf ", prev_u[i]);
        }
        printf("\n");

        write_hdf5_result(n1, n2, n3, execution_time, next_u);
    }

    close_hdf5_dataset(vel_model_id);
    close_hdf5_dataset(next_u_id);
    close_hdf5_dataset(prev_u_id);
    close_hdf5_dataset(coefficient_id);
    close_hdf5_file(file_id);

    #ifdef CONTAINS_MPI
        MPI_Finalize();
    #endif
}