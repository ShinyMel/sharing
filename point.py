import numpy as np
import hashlib

SCALE = 4

class Point:
    def __init__(self, coo, handler=False, name=None):
        self.name = name
        self.coo = coo
        self.handler = handler


    @property
    def coo(self):
        return self._coo
    
    @property
    def z(self):
        return complex( self._coo[0],  self._coo[1] )
    
    @coo.setter
    def coo(self, value):
        if isinstance(value, str):
            float_list = [float(x) for x in value.split(',')]
            coo = np.array(float_list)
        elif isinstance(value, complex):
            float_list = [value.real, value.imag]
            coo = np.array(float_list)
        elif isinstance(value, list):
            coo = np.array(value)
        elif isinstance(value, tuple):
            coo = np.array(list(value))
        elif isinstance(value, np.ndarray):
            coo = value
        else:
            raise ValueError("Data must be a string or a NumPy array")
        self._coo = np.round(coo, SCALE)


    @property
    def id(self):
        return f"{np.array2string(self._coo, precision=SCALE, floatmode='fixed')}"
 
    @property
    def real(self):
        return float(self.coo[0])
 
    @property
    def imag(self):
        return float(self.coo[1])
 
    def __array__(self):
        return self.coo

    def __list__(self):
        return self.coo

    def __repr__(self):
        h = "*" if {self.handler} else ""
        return f"{self.coo}{h}"
        #return f"Point(name={self.name}, coo={self.coo}, handler={self.handler})"
    
    def copy(self):
        return Point(self.coo, self.handler, self.name)
    
    def asstring(self):
        if self.handler:
            return   f"\x1B[0m{self.name}\x1B[0m{self.coo}"  
        else:
            return f"\x1B[32m{self.name}\x1B[0m{self.coo}"

    def tosvg(self):
        x, y = [ np.round(v,SCALE) for v in self.coo ]
        return f"{x},{y}"

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.value == other.value
        return False

    
