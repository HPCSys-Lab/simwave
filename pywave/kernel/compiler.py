import os

class Compiler():

    """
    Base class to implement the runtime compiler

    Parameters
    ----------
    program_version : str
        Specify the version of the program (sequential, cuda, openmp, openacc)
    c_code : str
        Path to the C code file
    """
    def __init__(self, program_version, c_code=None):
        self.version = program_version
        self.c_code = c_code
        self.flags = ['-O3', '-fPIC', '-Wall', '-std=c99', '-shared']

    def config_sequential(self):
        self.cc = 'gcc'

        if not self.c_code:
            self.c_code = 'sequential.c'

    def config_cuda(self):
        self.cc = 'nvcc'

        if not self.c_code:
            self.c_code = 'cuda.cu'

        self.flags.remove('-std=c99')
        self.flags.remove('-fPIC')
        self.flags.remove('-Wall')

        self.flags += ['-gencode arch=compute_75,code=sm_75', '-Xcompiler -fPIC']

    def config_openmp(self):
        self.cc = 'clang'

        if not self.c_code:
            self.c_code = 'openmp.c'

        self.flags.remove('-std=c99')
        self.flags += ['-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda', '-Xopenmp-target', '-march=sm_75', '-lm']

    def config_openacc(self):
        self.cc = 'pgcc'

        if not self.c_code:
            self.c_code = 'openacc.c'

        self.flags.remove('-std=c99')
        self.flags.remove('-O3')
        self.flags.remove('-Wall')
        self.flags += ['-fast', '-Minfo=all', '-ta=tesla']

    def compile(self, dimension='2d', density='constant_density', space_order_mode='multiple_space_order', operator='forward'):
        """
        Compile the program.

        Parameters
        ----------
        dimension : str
            Grid dimension. 2d or 3d.
        density : str
            Consider density or not. Options: constant (without density) or variable (consider density).
            Default is constant.
        space_order_mode: str
            Compile the version with multiple spatial orders (multiple) or the fixed second order version (fixed).
        operator : str
            Operator implementation. Only forward operator available at the moment.

        Returns
        ----------
        str
            Path to the compiled shared object
        """

        # get the working dir
        working_dir = os.getcwd()

        # get the dir of the compiler.py file
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # program dir
        program_dir = current_dir + '/c_code/{}/{}/{}/{}/'.format(
            operator, space_order_mode, density, dimension
        )

        object_dir = working_dir + '/tmp/'
        object_name = "lib_c_wave_{}.so".format(self.version.lower())

        if self.version == 'sequential':
            self.config_sequential()
        elif self.version == 'cuda':
            self.config_cuda()
        elif self.version == 'openmp':
            self.config_openmp()
        elif self.version == 'openacc':
            self.config_openacc()
        else:
            raise Exception('Program version (%s) unavailable' % self.version)

        cmd = self.cc + ' ' + program_dir + self.c_code + ' ' + ' '.join(self.flags) + ' -o ' + object_dir + object_name

        print('Compilation command:', cmd)

        # create a dir to save the compiled shared object
        os.makedirs(object_dir, exist_ok=True)

        # execute the command
        if os.system(cmd) != 0:
            raise Exception('Compilation failed')

        return object_dir + object_name
