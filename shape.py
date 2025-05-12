from collections import OrderedDict

import bezier
from pattern_crafting.point import Point
import numpy as np
import re

def _naming(key):
    if not isinstance(key, str):
        raise ValueError("Incorrect name", key)
    if not key[0].isalpha():
        raise ValueError("Incorrect name", key)
    return key

def _get_next_name(data, prefix="p"):
    for i in range(len(data)+1):
        name = f"{prefix}{i+1}"
        if not name in data:
            return name

class Shape(OrderedDict):
    def __init__(self, *args):
        self._closed = True
        if not args:
            super().__init__(*args)
        elif isinstance(args, tuple):
            d = { _naming(k):Point(v, name=k) if not isinstance(v,Point) else v.copy() for k,v in args[0].items()}
            super().__init__(d)
        else:
            raise ValueError("Dict expected")
            
    def __setitem__(self, key, value):
        key = _naming(key)
        value = Point(value) if not isinstance(value, Point) else value.copy()
        value.name = key
        super().__setitem__(key, value) 

    def __getitem__(self, key):
        return super().__getitem__(key)
   
    def __delitem__(self, key):
        super().__delitem__(key) 

    def reverse(self):
        l = list(self.items())[::-1]
        self.clear()
        self.update(l)

    def set_order(self,ordering):
        o = OrderedDict({k:self[k] for k in ordering.split(" ")})
        self.clear()
        self.update(o)
 
    def get_index(self,p):
        return  list(self.keys()).index(p)
   
    def to_string(self):
        return " ".join(self.keys())

    def rename(self, prefix):
        o = OrderedDict({f"{prefix}{i}":f"{i}>{v}" for i,v in enumerate(self.values())})
        self.clear()
        self.update(o)
  
  
    @property
    def closed(self):
        """
        bool: Indicates whether the object is in a closed state.

        This property returns the value of the `_closed` attribute, which indicates
        whether the object is considered closed.

        Returns:
            bool: The current state of the `_closed` attribute.

        Example:
            >>> self.closed
            True
        """
        return self._closed
    
    @closed.setter
    def closed(self, value):
        """
        Set the closed state of the object.

        This setter updates the `_closed` attribute to the given value.

        Args:
            value (bool): The new state to set for the `_closed` attribute.

        Example:
            >>> self.closed = False
        """
        self._closed = value


    def asstring(self):
        """
        Convert the object's points to a string representation.

        This method generates a string representation of all `Point` objects in the `_data` dictionary.
        It concatenates their string representations and appends a 'Z' if the object is closed.

        Returns:
            str: A string representation of the points and the closed state.

        Example:
            >>> self.asstring()
            'point1_asstring point2_asstring Z'

        Notes:
            - Only `Point` objects in the `_data` dictionary are included in the string.
            - If `self.closed` is True, a 'Z' is appended to the string.

        """
        return " ".join([v.asstring() for v in self.values() ]) + "  " + ("Z" if self.closed else "")



    def set_logical_path(self, value):
        """
        Set the logical path for the object based on a string value.

        This method parses a string representation of a path keys and updates the shape dictionary
        with `Point` objects corresponding to the path commands and coordinates.

        **Prerequisite**: The point names used in the path string should already exist in the shape.

        Args:
            value (str): A string representing the path in svg format, containing commands 
                        ('M', 'L', 'Q', 'C', 'Z') and coordinates.

        Example:
            When p1, p2, p3, p4, p5, p6, p7 should already exists in the shape
            >>> self.set_logical_path("M p1 L p2 Q p3 p4 C p5 p6 p7 Z")

        Notes:
            - The method uses regular expressions to clean and format the input string.
            - It iterates through the path commands and coordinates, updating the `_data` dictionary
            with `Point` objects.
            - The `closed` attribute is set to `True` if the 'Z' command is encountered.
            - The method handles different path commands ('M', 'L', 'Q', 'C') and sets the appropriate
            number of steps for each command.

        """
        newdata = {}
        step = 1
        value = re.sub(r'[\s\n]+', ' ', value.strip())
        value = re.sub(r'[\s\n]*,[\s\n]*', ',', value.strip())

        chunks_iter = iter(value.split(" "))
        while True:
            try:
                k = next(chunks_iter)
                m = 0

                if k in ("M", "L"):
                    step = 1    
                elif k == "Q":
                    step = 2
                elif k == "C":
                    step = 3
                elif k == "Z":
                    self.closed = True
                    break
                else:
                    m = 1
                    p = self[k].copy()
                    p.name = k
                    newdata[k] = p
                for _ in range(step - m):
                    k = next(chunks_iter)
                    p = self[k].copy()
                    p.name = k
                    newdata[k] = p
                    newdata[k].handler = True
                newdata[k].handler = False
            except StopIteration:
                break
        self.clear()
        self.update( newdata )

            














    def get_beads_names(self):
        """
        Generate sequences of names based on handler availability.

        This method iterates through the names and their shifted versions to yield
        pairs, triples, or quadruples of names where the corresponding objects do not
        have handlers.

        Returns:
            generator: A generator that yields tuples of names (pairs, triples, or quadruples)
                    based on the presence of handlers in the subsequent objects.

        Example:
            >>> list(self.get_beads_names())
            [('p1', 'p2'), ('p2', 'p3', 'p4'), ('p4', 'p1')]

        Notes:
            - The method uses a circular shift of the names list if `self.closed` is True.
            - If `self.closed` is False, a placeholder `"-"` is added to the end of the shifted list.
            - The method checks for the presence of handlers in the objects corresponding to the names.
            - It yields tuples of names where the objects do not have handlers.

        """
        names = list(self.keys()) 
        if self.closed:
            names2 = names[1:] + [names[0]]
        else:
            names2 = names[1:] + ["-"]
        for n1, n2, n3, n4 in zip(names, names2, names[2:] + ["-", "-"], names[3:] + ["-", "-", "-"]):
            p1, p2, p3, p4 = [self.get(n) for n in (n1, n2, n3, n4)]
            if not p1.handler:
                if p2 and not p2.handler:
                    yield (n1, n2)
                elif p3 and not p3.handler:
                    yield (n1, n2, n3)
                elif p4 and not p4.handler:
                    yield (n1, n2, n3, n4)

    def get_beads_items(self):
        """
        Generate sequences of names and points based on handler availability.

        This method iterates through the names and their shifted versions to yield
        pairs, triples, or quadruples of names where the corresponding objects do not
        have handlers.

        Returns:
            generator: A generator that yields tuples of tuple of name and point (pairs, triples, or quadruples)
                    based on the presence of handlers in the subsequent objects.

        Notes:
            - The method uses a circular shift of the names list if `self.closed` is True.
            - If `self.closed` is False, a placeholder `"-"` is added to the end of the shifted list.
            - The method checks for the presence of handlers in the objects corresponding to the names.
            - It yields tuples of names where the objects do not have handlers.

        """
        names = list(self.keys()) 
        if self.closed:
            names2 = names[1:] + [names[0]]
        else:
            names2 = names[1:] + ["-"]
        for n1, n2, n3, n4 in zip(names, names2, names[2:] + ["-", "-"], names[3:] + ["-", "-", "-"]):
            p1, p2, p3, p4 = [self.get(n) for n in (n1, n2, n3, n4)]
            if not p1.handler:
                if p2 and not p2.handler:
                    yield ( (n1, self[n1]),  (n2, self[n2]))
                elif p3 and not p3.handler:
                    yield ( (n1, self[n1]),  (n2, self[n2]), (n3, self[n3]))
                elif p4 and not p4.handler:
                    yield ( (n1, self[n1]),  (n2, self[n2]), (n3, self[n3]), (n4, self[n4]))


    def get_svg_path_d_keys(self):
        """
        Generate the SVG path 'd' attribute keys.

        This method constructs the 'd' attribute for an SVG path element by iterating
        through the beads and generating the appropriate SVG path commands.

        Returns:
            str: A string representing the 'd' attribute of an SVG path.

        Example:
            >>> self.get_svg_path_d_keys()
            'M p0 L p1 Q p2 p3 C p4 p5 p6 Z'

        Notes:
            - The method uses the `get_beads` method to retrieve sequences of names.
            - It uses different SVG path commands ('M', 'L', 'Q', 'C') based on the length of the bead sequences.
            - If `self.closed` is True, the path is closed with a 'Z' command.
            - The method ensures that consecutive commands are not repeated unnecessarily.

        """
        typos = ["M", "L", "Q", "C"]
        path = ["M"]
        typo = ""
        
        beads = list(self.get_beads_names())
        if self.closed:
            beads.pop()
        for b in beads:
            b = list(b)
            i = len(b) - 1
            f = b.pop(0)
            newtypo = typos[i]
            if f != path[-1]:
                path.append(f)
            if newtypo != typo:
                path.append(newtypo)
                typo = newtypo
            path.extend(b)
        if self.closed:
            path.append("Z")
        return " ".join(path)

    def get_svg_path_d(self):
        """
        Generate the SVG path 'd' attribute for the shape.

        This function constructs the 'd' attribute for an SVG path element based on the points (beads) of the shape.
        It uses different path commands ('M', 'L', 'Q', 'C') depending on the number of points in each segment.

        Returns:
        str: The 'd' attribute for the SVG path element.
        """
        typos = ["M", "L", "Q", "C"]
        path = ["M"]
        typo = ""

        pts = list(self.values())

        if len(pts) == 0 :
            return "M 0,0"
        
        if len(pts) == 1 :
            f = pts[0]
            return f"M {f.tosvg()}"

        beads = list(self.get_beads_names())
        
        if self.closed:
            beads.pop()
        l = ""
        for b in beads:
            b = list(b)
            i = len(b) - 1
            f = b.pop(0)
            newtypo = typos[i]
            if f != l:
                path.append(self[f].tosvg())
            if newtypo != typo:
                path.append(newtypo)
                typo = newtypo
            path.extend([self[x].tosvg() for x in b])
            l = b[-1]
        if self.closed:
            path.append("Z")
        return " ".join(path)

    def set_svg_path_d(self, path, prefix = "p"):
        data = {}
        maps = { "M":1, "L":1, "Q":2, "C":3 }
        # Remove unnecessary whitespace and newlines
        spath = re.sub(r'[\s\n]+', ' ', path.strip())
        spath = re.sub(r'[\s\n]*,[\s\n]*', ',', spath.strip())
        
        # Split the path into a list of commands and coordinates
        step = 0
        for p in spath.split():
            if p in maps :
                step = maps[p]
                i = 0
            elif p == "Z" :
                self.closed = True
            else:
                i = i +1
                name = _get_next_name(data, prefix)
                handler = not (i == step)
                data[name] = Point(p, handler=handler)
                if not handler:
                    i = 0 
                
        self.clear()
        self.update(data)
    

    def get_bezierpkg_curves(self):
        names = list(self.keys()) 
    
        if self.closed:
            names2 = names[1:] + [names[0]]
        else:
            names2 = names[1:] + ["-"]

        for n1, n2, n3, n4 in zip(names, names2, names[2:] + ["-", "-"], names[3:] + ["-", "-", "-"]):
            p1, p2, p3, p4 = [self.get(n) for n in (n1, n2, n3, n4)]
            
            if not p1.handler:
                if p2 and not p2.handler:
                    points = np.array([ p.coo for p in (p1, p2)])
                    if np.allclose(p1, p2):
                        continue
                elif p3 and not p3.handler:
                    points = np.array([ p.coo for p in (p1, p2, p3)])
                elif p4 and not p4.handler:
                    points = np.array([ p.coo for p in (p1, p2, p3, p4)])
                
                yield bezier.Curve.from_nodes(points.T)
