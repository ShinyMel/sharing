from typing import List

try:
    import numpy as np
except ImportError:
    print("Cannot import numpy")
    np = None

try:
    import bezier as bz
except ImportError:
    print("Cannot import bezier")
    bz = None


def np_to_complex(a):
    [x], [y] = a.tolist()
    return complex(x, y)


def bezier_to_complex(curve: object) -> List[complex]:
    if hasattr(curve, "nodes"):
        return [complex(x, y) for x, y in curve.nodes.T]
    raise ValueError("invalid input type. bezier.curve expected")


def complex_to_bezier(control_points: List[complex]) -> object:
    if not bz:
        raise ImportError(
            "The optional module bezier could not be imported. Please ensure it is installed."
        )

    # Convert complex numbers into a NumPy array (fortran order expected)
    nodes = np.asfortranarray(
        [
            [point.real for point in control_points],
            [point.imag for point in control_points],
        ]
    )

    # Create a Bézier curve
    return bz.Curve(nodes, degree=len(control_points) - 1)


def bezier_to_svg(curve: object) -> str:
    l = bezier_to_complex(curve)
    n = len(l)
    if not 2 <= n <= 4:
        raise ValueError("Expecting a ")

    if len(l) == 2:
        x1, y1, x2, y2 = l[0].real, l[0].imag, l[1].real, l[1].imag
        return f"""<line x1="{x1}" y1="{y1}" x2="{x2}" y1="{y2}" />"""
    l = [f"""{z.real},{z.imag}""" for z in l]
    l.insert(0, "M")
    if n == 3:
        l.insert(2, "Q")
    elif n == 4:
        l.insert(2, "C")

    return " ".join(l)


def init_parse(values):
    if isinstance(values, list):
        if isinstance(values[0], complex):
            l = [v for v in values]
        else:
            l = [float(v) for v in values]
    elif hasattr(values, "nodes"):
        l = [complex(x, y) for x, y in values.nodes.T]
    elif np and isinstance(values, np.ndarray):
        l = [complex(x, y) for x, y in values.T]
    return l

class BezierBase(object):


 

    @property
    def area(self):
        pts = [self.evaluate(t) for t in np.linspace(0, 1, 1000)]
        if max([np.abs(z) for z in pts]) > 10e6:
            raise ValueError("Can't compute area")
    
        xs = [z.real for z in pts]
        ys = [z.imag for z in pts]
            
        area = 1 / 2 * np.sum(xs * np.roll(ys, 1) - ys * np.roll(xs, 1))
        return area

    def get_polynomial_coefficients(self):
        raise NotImplementedError("Subclasses must implement this method")


    def get_derivative_polynomial_coefficients(self):
        raise NotImplementedError("Subclasses must implement this method")


    def evaluate(self, t):
        raise NotImplementedError("Subclasses must implement this method")


    def evaluate_derivative(self, t):
        raise NotImplementedError("Subclasses must implement this method")


    def evaluate_second_derivative(self, t):
        raise NotImplementedError("Subclasses must implement this method")


    def derivative_roots(self):
        raise NotImplementedError("derivative_roots")

    def x_aligned(self):
        raise NotImplementedError("Subclasses must implement this method")


    def unit_referentiel(self, t):
        vt = self.evaluate_derivative(t)
        vt = vt / abs(vt)
        vn = vt * np.exp(-1j * np.pi / 2)
        return vt, vn


    def evaluate_curvature(self, t):
        raise NotImplementedError("Subclasses must implement this method")


    def as_bezier(self):
        # Convert complex numbers into a NumPy array (fortran order expected)
        nodes = np.asfortranarray(
            [
                [point.real for point in self._control_points],
                [point.imag for point in self._control_points],
            ]
        )
        # Create a Bézier curve
        if bz:
            return bz.Curve(nodes, degree=self._degree)
        else:
            return nodes

    def specialized(self, t1, t2):
        return self.__class__()(self.as_bezier().specialize(t1, t2))

    @property
    def apex(self):
        sb = self.x_aligned()
        ys = [z.imag for z in sb.get_derivative_polynomial_coefficients()]
        ts = [ t for t in np.roots(ys) if 0<t<1 ]
        if not ts:
            return( (sb._control_points[0] + sb._control_points[-1])/2 )
        #return [(float(t), float(sb.evaluate(t).imag), self.evaluate(t)) for t in ts]
        rootpts = [sb.evaluate(t) for t in ts]
        i = np.argmax([np.abs(z.imag) for z in rootpts])
        return self.evaluate(ts[i])

