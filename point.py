import numpy as np
import hashlib
import numpy as np


class Point(complex): 

    def __new__(cls, arg1, arg2=None, handler=None, invert_y = False):
        real, imag = arg1, arg2

        if isinstance(arg1, Point):
            real, imag, handler = arg1.real, arg1.imag, handler or arg1.handler

        elif isinstance(arg1, complex):
            real, imag = arg1.real, arg1.imag
        elif isinstance(arg1, np.ndarray):
            real, imag = real[0], real[1]
        
        elif isinstance(arg1, str):
            real, imag = [float(x) for x in real.split(',')]
        
        elif isinstance(arg1, (list,tuple)):
            real, imag = [float(x) for x in arg1]
    
        if invert_y:
            imag = -imag

        if handler is None:
            handler = False
        real, imag = np.round(real,6), np.round(imag,6)
         
        instance = super().__new__(cls, real, imag)
        instance.handler = handler
        return instance

 
    def __repr__(self):
        if self.handler:
            return f"~({self.real},{self.imag})"
        else:
            return f"({self.real},{self.imag})"
    
    def __str__(self ):
        if self.handler:
            return   f"\x1B[0m{self.real},{self.imag}\x1B[0m"  #default
        else:
            return f"\x1B[32m{self.real},{self.imag}\x1B[0m" #green
    
    @property
    def svg(self):
        imag =  -self.imag
        if self.handler:
            return   f"\x1B[0m{self.real},{imag}\x1B[0m"  #default
        else:
            return f"\x1B[32m{self.real},{imag}\x1B[0m" #green
    
    @staticmethod
    def from_svg( svg, handler=False):
        return Point(svg, handler=handler, invert_y=True)
 
    def __array__(self):
        return np.array([self.real, self.imag])

    def __list__(self):
        return [self.real, self.imag]

    def __complex__(self):
        return complex(self.real, self.imag)  # Converts to a complex number
    
    def copy(self):
        return Point(self.real, self.imag, self.handler)
    
    def asstring(self):
        if self.handler:
            return   f"\x1B[0m{self.name}\x1B[0m{self.coo}"  
        else:
            return f"\x1B[32m{self.name}\x1B[0m{self.coo}"

    def __eq__(self, other):
        if isinstance(other, complex):
            return self.real == other.real and self.imag == other.imag  
        if isinstance(other, Point):
            return self.real == other.real and self.imag == other.imag and self.handler == other.handler 
        raise ValueError("Unexpected Type")
