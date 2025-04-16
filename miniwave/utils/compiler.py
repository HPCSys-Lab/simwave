import os
from hashlib import sha1
from typing import Optional
from utils.properties import Properties

class Compiler:
    """
    Base class to implement the runtime compiler.
    
    Parameters
    ----------
    language : str
        Kernel language.    
    sm : int
        Cuda capability. 
    fast_math : bool
        Enable fast_math. Default is False. 
    """
    
    def __init__(self, language: str, sm: int, fast_math: Optional[bool] = False):
        self._language = language      
        self._sm = sm
        self._fast_math = fast_math
        
        self._define_default_flags()
        
    @property
    def language(self) -> str:
        return self._language
    
    @property
    def sm(self) -> int:
        return self._sm
    
    @property
    def fast_math(self) -> bool:
        return self._fast_math      

    @property
    def cc(self) -> str:
        return self._cc
    
    @cc.setter
    def cc(self, value: str) -> None:
        self._cc = value
    
    @property
    def flags(self) -> str:
        return self._flags
    
    @flags.setter
    def flags(self, value: str) -> None:
        self._flags  = value  
        
    def _define_default_flags(self) -> None:
        
        # fast math for GNU and CLANG
        if self.fast_math:
            fast_math = "-ffast-math"
        else:
            fast_math = ""               
    
        if self.language == 'c':
            self.cc = 'gcc'
            self.flags = f'-O3 -fPIC {fast_math} -shared'
            
        elif self.language == 'openmp':
            self.cc = 'clang'
            self.flags = f'-O3 -fPIC {fast_math} -fopenmp -fopenmp-version=51 -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_{self.sm} -shared -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -g'

        elif self.language == 'mpi':
            self.cc = 'mpicc'
            self.flags = f'-O3 -fPIC {fast_math} -shared'

        elif self.language == 'mpi_cuda':
            self.cc = 'mpicc'
            self.flags = f'-O3 -fPIC {fast_math} -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include --offload-arch=sm_{self.sm} -shared'
            
        elif self.language == 'openmp_cpu':
            self.cc = 'gcc'
            self.flags = f'-O3 -fPIC {fast_math} -fopenmp -shared'
            
        elif self.language == 'openacc':
            
            if self.fast_math:
                fast_math = ",fastmath" 
            
            self.cc = 'pgcc'
            self.flags = f'-O3 -fPIC -acc:gpu -gpu=pinned{fast_math} -shared'            
            
        elif self.language == 'cuda':            
            if self.fast_math:
                fast_math = "--use_fast_math"
                
            self.cc = 'nvcc'
            self.flags = f'-O3 -gencode arch=compute_{self.sm},code=sm_{self.sm} --compiler-options -fPIC,-Wall {fast_math} -shared'
        
        elif self.language == 'python':
            pass

        elif self.language == 'ompc':
            self.cc = 'clang'
            self.flags = f'-O3 -fPIC {fast_math} -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu -shared'
        
        else:
            raise Exception("Language not supported")
    
    def compile(self, file: str, properties: Optional[Properties] = None) -> str:
        
        object_dir = "/tmp/miniwave/"
        
        # create a dir to save the compiled shared object
        os.makedirs(object_dir, exist_ok=True)
        
        # get c file content
        with open(file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # float precision  
        type = {
            'float32': '-DFLOAT',
            'float64': '-DDOUBLE'
        }
        
        float_precision = type[properties.dtype]
            
        b1, b2, b3 = properties.block3
        stencil_radius = properties.stencil_radius
        
        c_def = f" -DBLOCK1={b1} -DBLOCK2={b2} -DBLOCK3={b3} -DSTENCIL_RADIUS={stencil_radius} {float_precision}"
        
        # add defined properties to flags 
        self.flags += c_def
        
        # compose the object string
        object_str = "{} {} {}".format(self.cc, self.flags, file_content)
        
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
            cmd = self.cc + " " + file + " " + self.flags + " -o " + object_path
            
            print("Compilation command:", cmd)
            
            # execute the command
            if os.system(cmd) != 0:
                raise Exception("Compilation failed")
            
        return object_path
