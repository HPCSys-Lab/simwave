import os


class Compiler:
    """
    Base class to implement the runtime compiler.

    Parameters
    ----------
    cc : str, optional
        C compiler. Default is 'gcc'.
    cflags : str, optional
        C compiler flags. Default is '-O3 -fPIC -Wall -std=c99 -shared'.
    """

    def __init__(self, cc="gcc", cflags="-O3 -fPIC -Wall -std=c99 -shared"):
        self.cc = cc
        self.cflags = cflags

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
        space_order_mode: str
            Compile the version with multiple spatial orders (multiple)
            or the fixed second order version (fixed).
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

        # program dir
        program_dir = current_dir + "/c_code/{}/{}/{}d/".format(
            operator, density, dimension
        )

        object_dir = working_dir + "/tmp/"
        c_code_name = "sequential.c"
        object_name = "lib_c_wave-{}d.so".format(dimension)

        cmd = (
            self.cc
            + " "
            + program_dir
            + c_code_name
            + " "
            + self.cflags
            + " -o "
            + object_dir
            + object_name
            + " {}".format(float_precision)
        )

        print("Compilation command:", cmd)

        # create a dir to save the compiled shared object
        os.makedirs(object_dir, exist_ok=True)

        # execute the command
        if os.system(cmd) != 0:
            raise Exception("Compilation failed")

        return object_dir + object_name
