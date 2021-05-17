import os
from hashlib import sha1


class Compiler:
    """
    Base class to implement the runtime compiler.

    Parameters
    ----------
    cc : str, optional
        C compiler. Default is gcc.
    language: str, optional
        Define the code implementation language like: c (sequential),
        cpu_openmp (parallel CPU). Default is c.
    cflags : str, optional
        C compiler flags.
        Default is '-O3 -fPIC -Wall -std=c99 -shared'.
    """
    def __init__(self, cc='gcc', language='c', cflags=None):
        self.cc = cc
        self.language = language
        self.cflags = cflags

    @property
    def cc(self):
        return self._cc

    @cc.setter
    def cc(self, value):
        if not isinstance(value, str):
            raise TypeError("Compiler.cc attribute must be str.")

        self._cc = value

    @property
    def cflags(self):
        return self._cflags

    @cflags.setter
    def cflags(self, value):
        # use default flags
        if value is None:
            value = '-O3 -fPIC -Wall -std=c99 -shared'

        if not isinstance(value, str):
            raise TypeError("Compiler.cflags attribute must be str.")

        # add -shared flag if not provided
        if '-shared' not in value:
            value += ' -shared'

        # add OpenMP flag if it is not provided and version is cpu_openmp
        if self.language in ('cpu_openmp', 'gpu_openmp'):
            omp_flag = self.get_openmp_flag()

            if omp_flag is None:
                print("WARNING: make sure OpenMP flag is provided in cflags.")
            else:
                if omp_flag not in value:
                    value += ' {}'.format(omp_flag)

        self._cflags = value

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        if not isinstance(value, str):
            raise TypeError("Compiler.language attribute must be str.")

        options = ['c', 'cpu_openmp', 'gpu_openmp']

        if value not in options:
            raise ValueError(
                "Compiler.language {} not implemented.".format(value)
            )

        self._language = value

    def get_openmp_flag(self):
        """
        Get the OpenMP flag according the compiler.
        Returns None if compiler is unknown.

        Returns
        ----------
        str
            OpenMP flag's name for compilation command.
        """
        omp_flag = {
            'gcc': '-fopenmp',
            'icc': '-openmp',
            'pgcc': '-mp',
            'clang': '-fopenmp'
        }

        return omp_flag.get(self.cc)

    def compile(self, dimension, density, float_precision, operator):
        """
        Compile the program.

        Parameters
        ----------
        dimension : int
            Grid dimension. 2D (2) or 3D (3).
        density : str
            Consider density or not. Options: constant (without density)
            or variable (consider density). Default is constant.
        float_precision : str
            Float single (C float) or double (C double) precision.
        operator : str
            Operator implementation.
            Only forward operator available at the moment.

        Returns
        ----------
        str
            Path to the compiled shared object
        """
        # get the working dir
        working_dir = os.getcwd()

        # get the dir of the compiler.py file
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # c program root dir
        program_dir = current_dir + "/c_code/{}/{}/{}d/".format(
            operator, density, dimension
        )

        # c pragram file name
        c_code_name = "wave.c"

        # c code complete path
        program_path = program_dir + c_code_name

        # get c file content
        with open(program_path, 'r') as f:
            c_file_content = f.read()

        # object root dir
        object_dir = working_dir + "/tmp/"

        # define the language
        if self.language == 'cpu_openmp':
            language_c = ' -DCPU_OPENMP'
        elif self.language == 'gpu_openmp':
            language_c = ' -DGPU_OPENMP'
        elif self.language == 'c':
            language_c = ''

        # compose the object string
        object_str = "wave {} {} {} {}d {} {} {} {} \n{}".format(
            self.language,
            self.cc,
            self.cflags,
            dimension,
            operator,
            density,
            float_precision,
            language_c,
            c_file_content
        )

        # apply sha1 hash to name the object
        hash = sha1()
        hash.update(object_str.encode())
        object_name = hash.hexdigest() + ".so"

        # object complete path
        object_path = object_dir + object_name

        # check if object_file already exists
        if os.path.exists(object_path):
            print("Shared object already compiled in:", object_path)
        else:
            cmd = (
                self.cc
                + " "
                + program_path
                + " "
                + self.cflags
                + " {}".format(float_precision)
                + language_c
                + " -o "
                + object_path
            )

            print("Compilation command:", cmd)

            # create a dir to save the compiled shared object
            os.makedirs(object_dir, exist_ok=True)

            # execute the command
            if os.system(cmd) != 0:
                raise Exception("Compilation failed")

        return object_path
