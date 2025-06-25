from collections import OrderedDict
from dataclasses import dataclass
import numbers
import re

def is_numeric_but_not_bool(value):
    return isinstance(value, numbers.Number) and not isinstance(value, bool)



# Define the regex pattern
namepattern = r'^[A-Za-z]\d+'


class Handle:
    def __init__(self, coord, is_smooth=False):
        self._coord = complex( coord )
        self._is_smooth = bool( is_smooth )

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        self._coord = complex(value)

    @property
    def is_smooth(self):
        return self._is_smooth

    @is_smooth.setter
    def is_smooth(self, value):
        self._is_smooth = bool(value)

    def __repr__(self):
        return f"Handle(coord={self.coord}, is_smooth={self.is_smooth})"



class ShapeBase(OrderedDict):
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if name == "closed":
                object.__setattr__(self, "_closed", value)
            else:
                if re.search(namepattern, name):
                    if is_numeric_but_not_bool(value) :
                        self.__dict__[name] = Handle(value, False)
                    elif isinstance(value, bool) :
                        self.__dict__[name] = Handle(0, value)
                    elif isinstance(value, tuple) :
                        c,s = value
                        self.__dict__[name] = Handle(c,s)
                    else:
                        raise ValueError("Unexpected value")
    @property
    def closed(self):
        return self._closed
    
    @closed.setter
    def closed(self, value):
        self._closed = value
    
    def __setattr__(self, name, value):
        if name == 'closed':
            object.__setattr__(self, "_closed", value)
        elif re.search(namepattern, name):
            if is_numeric_but_not_bool(value) :
                self.__dict__[name] = Handle(value, False)
            elif isinstance(value, bool) :
                self.__dict__[name] = Handle(0, value)
            elif isinstance(value, tuple) :
                c,s = value
                self.__dict__[name] = Handle(c,s)
            else:
                raise ValueError("Unexpected value")

    def reorder(self, order):
        if not all(name in self.__dict__ for name in order):
            raise ValueError("All names in the order list must be keys in the dictionary")
        reordered_dict = OrderedDict((name, self.__dict__[name]) for name in order)
        self.__dict__.clear()
        self.__dict__.update(reordered_dict)

    def __repr__(self):
        return "ShapeBase:" + "\n  " + "\n  ".join( f"{k}={v}" for k,v in self.__dict__.items())


class Shape(ShapeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __repr__(self):
        return "Shape:" + "\n  " + "\n  ".join( f"{k}={v}" for k,v in self.__dict__.items())

    
obj = Shape(closed=True, p1=101j, p2=102j)
obj.closed = False
obj.k1 = 55
obj.k1.is_smooth = True
print("k1", obj.k1)
print("closed", obj.closed)
obj.closed = True
print("closed",obj.closed)
obj.k1.coord = 1j
print("k1", obj.k1)
obj.k2 = (2j,True)
print("k2", obj.k2)
obj.k3 = 3
obj.k4 = 4+4j
print(obj)
obj.k5 = 69

obj.reorder(['k5','k4','k2','k1', 'k3'])
print(obj)

