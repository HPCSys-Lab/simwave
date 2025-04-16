
// use single (float) or double precision
// according to the value passed in the compilation cmd
#if defined(FLOAT)
    typedef float f_type;
#elif defined(DOUBLE)
    typedef double f_type;
#else
    typedef float f_type;
#endif

int forward(f_type *prev_u, f_type *next_u, f_type *vel_model, f_type *coefficient,
            f_type d1, f_type d2, f_type d3, f_type dt, int n1, int n2, int n3,
            int iterations, int stencil_radius,
            int block_size_1, int block_size_2, int block_size_3
           );