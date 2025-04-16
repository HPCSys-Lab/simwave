#include <stdio.h>
#include <stdlib.h>

#include "sequential.h"

int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3
           ){

    f_type d1Squared = d1 * d1;
    f_type d2Squared = d2 * d2;
    f_type d3Squared = d3 * d3;
    f_type dtSquared = dt * dt;

    for(int t = 0; t < iterations; t++) {     
        
        for(int i = stencil_radius; i < n1 - stencil_radius; i++){
            for(int j = stencil_radius; j < n2 - stencil_radius; j++){
                for(int k = stencil_radius; k < n3 - stencil_radius; k++){
                    // index of the current point in the grid
                    int current = (i * n2 + j) * n3 + k;

                    // stencil code to update grid
                    f_type value = coefficient[0] * (prev_u[current]/d1Squared + prev_u[current]/d2Squared + prev_u[current]/d3Squared);

                    // radius of the stencil                    
                    for(int ir = 1; ir <= stencil_radius; ir++){
                        value += coefficient[ir] * (
                                ( (prev_u[current + ir] + prev_u[current - ir]) / d3Squared ) + //neighbors in the third axis
                                ( (prev_u[current + (ir * n3)] + prev_u[current - (ir * n3)]) / d2Squared ) + //neighbors in the second axis
                                ( (prev_u[current + (ir * n2 * n3)] + prev_u[current - (ir * n2 * n3)]) / d1Squared )); //neighbors in the first axis
                    }

                    value *= dtSquared * vel_model[current] * vel_model[current];
                    next_u[current] = 2.0 * prev_u[current] - next_u[current] + value;
                }
            }
        }

        // swap arrays for next iteration
        f_type *swap = next_u;
        next_u = prev_u;
        prev_u = swap;        
    }

    return 0;
}
