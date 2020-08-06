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
            self.c_code = 'forward_sequential.c'

    def config_cuda(self):
        self.cc = 'nvcc'

        if not self.c_code:
            self.c_code = 'forward_cuda.cu'

        self.flags.remove('-std=c99')
        self.flags.remove('-fPIC')
        self.flags.remove('-Wall')

    def config_openmp(self):
        self.cc = 'clang'

        if not self.c_code:
            self.c_code = 'forward_openmp.c'

        self.flags.remove('-std=c99')
        self.flags += ['-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda', '-Xopenmp-target', '-march=sm_75', '-lm']

    def config_openacc(self):
        self.cc = 'pgcc'

        if not self.c_code:
            self.c_code = 'forward_openacc.c'

        self.flags.remove('-std=c99')
        self.flags.remove('-O3')
        self.flags.remove('-Wall')
        self.flags += ['-fast', '-Minfo', '-ta=tesla:managed' '-acc']

    """
    Compile the program

    Returns
    ----------
    str
        Path to the compiled shared object
    """
    def compile(self):

        current_dir = os.getcwd()
        object_dir = current_dir + '/tmp/'
        object_name = "lib_c_wave_{}.so".format(self.version.lower())
        program_dir = current_dir + '/c_code/'

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
