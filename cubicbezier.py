import os
from typing import List
from bezierbase import BezierBase, init_parse
from scipy.signal import argrelextrema
import cmath
import numpy as np
#import bezier as bz

# try:
#     import numpy as np
# except ImportError:
#     print("Cannot import numpy")
#     np = None

# try:
#     import bezier as bz
# except ImportError:
#     print("Cannot import bezier")
#     bz = None


# Set error handling to raise exceptions
np.seterr(over='raise', under='raise', divide='raise', invalid='raise')


# def np_to_complex(a):
#     [x], [y] = a.tolist()
#     return complex(x, y)


# def bezier_to_complex(curve: object) -> List[complex]:
#     if hasattr(curve, "nodes"):
#         return [complex(x, y) for x, y in curve.nodes.T]
#     raise ValueError("invalid input type. bezier.curve expected")


# def complex_to_bezier(control_points: List[complex]) -> object:
#     if not bz:
#         raise ImportError(
#             "The optional module bezier could not be imported. Please ensure it is installed."
#         )

#     # Convert complex numbers into a NumPy array (fortran order expected)
#     nodes = np.asfortranarray(
#         [
#             [point.real for point in control_points],
#             [point.imag for point in control_points],
#         ]
#     )

#     # Create a Bézier curve
#     return bz.Curve(nodes, degree=len(control_points) - 1)


# def bezier_to_svg(curve: object) -> str:
#     l = bezier_to_complex(curve)
#     n = len(l)
#     if not 2 <= n <= 4:
#         raise ValueError("Expecting a ")

#     if len(l) == 2:
#         x1, y1, x2, y2 = l[0].real, l[0].imag, l[1].real, l[1].imag
#         return f"""<line x1="{x1}" y1="{y1}" x2="{x2}" y1="{y2}" />"""
#     l = [f"""{z.real},{z.imag}""" for z in l]
#     l.insert(0, "M")
#     if n == 3:
#         l.insert(2, "Q")
#     elif n == 4:
#         l.insert(2, "C")

#     return " ".join(l)



def center_multiple(points):
    # Calcul des différences
        P1 = points[:-2]
        P2 = points[1:-1]
        P3 = points[2:]
 
        #  (x1, y1), (x2, y2), (x3, y3) = A, B, C
        x1, y1 = P1.real, P1.imag
        x2, y2 = P2.real, P2.imag
        x3, y3 = P3.real, P3.imag
        

        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        A[abs(A)<.00001] = 0.00001
        B = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
        C = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
        return (-B / A / 2) + 1j* ( -C / A / 2)



def line_intersection(p1, v1, p2, v2):
    """
    Trouve l'intersection de deux lignes définies par un point et un vecteur en utilisant des nombres complexes.
    
    p1, p2 : nombres complexes
        Points par lesquels passent les lignes.
    v1, v2 : nombres complexes
        Vecteurs directeurs des lignes.
    
    Retourne :
    Nombre complexe représentant le point d'intersection, ou None si les lignes sont parallèles.
    """
    # Calcul des déterminants
    det = v1.real * v2.imag - v1.imag * v2.real
    if det == 0:
        # Les lignes sont parallèles
        return None
    
    # Calcul des paramètres t et s
    t = ((p2 - p1).real * v2.imag - (p2 - p1).imag * v2.real) / det
    intersection = p1 + t * v1
    return intersection
 



def solve(p1, p4, dp1, dp4, apex):
    try:
        q1, q4 = p1, p4
        
        apex = apex-p1
        p4 = p4-p1
        p1 = p1-p1

        a = np.angle( p4 )
        r = np.exp(-1j*a)

        u = dp1 / abs(dp1)
        v = dp4 / abs(dp4)
        rapex =  apex * r
        ru = u * r
        rv = v * r

        t1 = (( 2 * ru.imag - rv.imag) + np.sqrt( ru.imag **2 -ru.imag * rv.imag + rv.imag**2)) / (3 * (ru.imag-rv.imag))
        t2 = (( 2 * ru.imag - rv.imag) - np.sqrt( ru.imag **2 -ru.imag * rv.imag + rv.imag**2)) / (3 * (ru.imag-rv.imag))
        t = [ t for t in (t1, t2) if 0<t<1][0]

        alpha = rapex.imag / (3*t*(ru.imag + t**2 * (ru.imag-rv.imag) + t * (-2*ru.imag + rv.imag)))
        p2 = q1 + u*alpha
        p3 = q4 + v*alpha
        return p2, p3

    except:
        q = line_intersection(p1, dp1, p4, dp4)
        if q is None:
            return (q1*3+q4)/4, (q1+q4*3)/4
        return (q1+q)/2, (q+q4)/2



def _sub_offset_solve(curve, ts, newpoints, i1, i2):   

    pts =newpoints[i1: i2]
    t1, t2 = ts[i1], ts[i2]
    q1 = pts[0]
    q4 = pts[-1]
    v = q4-q1
    rot = cmath.exp( -1j * cmath.phase(v))

    if i2-i1 <1:
        q2 = (q1*3+q4)/4
        q3 = (q1+q4*3)/4
        apexidx = i1
        apex = (q1+q2)/2
        c = CubicBezier([q1, q2, q3, q4])
        return apex, apexidx, c
        


    magnitudes = list(abs(z.imag) for z in [ (z-q1)*rot for z in pts ]) 
    i = np.argmax(magnitudes)
    apexidx = i1+i
    apex = newpoints[apexidx]


    dq1 = curve.evaluate_derivative(t1)
    dq4 = -curve.evaluate_derivative(t2)
    q2, q3 = solve(q1, q4, dq1, dq4, apex)
    ref = abs(q4-q1)
    if abs(q2-q1)>ref or abs(q3-q4)>ref or abs(q2-q3)>ref:
        q2 = (q1*3+q4)/4
        q3 = (q1+q4*3)/4
        
    c = CubicBezier([q1, q2, q3, q4])
    return apex, apexidx, c

 

class CubicBezier(object):
    def __init__(self, values):

        l = init_parse(values)
        self._degree = len(l) - 1
        self._control_points = l
    
        if self._degree == 0:
            raise ValueError("It is a plot (1 control point). Cubic (4 control points) expected")
        if self._degree == 1:
            raise ValueError("It is a segment (2 control points). Cubic (4 control points) expected")
        if self._degree == 2:
             raise ValueError("It is a quadratic curve (3 control points). Cubic (4 control points) expected")
        if self._degree == 3:
            p0, p1, p2, p3 = values
            # cubic
            # C = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) *t**2 * p2 + t**3 *p3
            # C = -p0*t**3 + 3*p0*t**2 - 3*p0*t + p0 + 3*p1*t**3 - 6*p1*t**2 + 3*p1*t - 3*p2*t**3 + 3*p2*t**2 + p3*t**3
            # C = (-p0 + 3*p1- 3*p2 + p3) * t**3  + ( 3*p0 - 6*p1 + 3*p2) *t**2 + (-3*p0 + 3*p1 )*t + p0
            # H = (-3*p0 + 9*p1 - 9*p2 + 3*p3)*t**2  + ( 6*p0 - 12*p1 + 6*p2) *t + (- 3*p0  + 3*p1)
            # H' = 2*(-3*p0 + 9*p1 - 9*p2 + 3*p3)*t  + ( 6*p0 - 12*p1 + 6*p2)

            self._C = [
                (-p0 + 3 * p1 - 3 * p2 + p3),
                (3 * p0 - 6 * p1 + 3 * p2),
                (-3 * p0 + 3 * p1),
                p0,
            ]
            self._D = [
                (-3 * p0 + 9 * p1 - 9 * p2 + 3 * p3),
                (6 * p0 - 12 * p1 + 6 * p2),
                (-3 * p0 + 3 * p1),
            ]
            self._S = [
                2 * (-3 * p0 + 9 * p1 - 9 * p2 + 3 * p3),
                (6 * p0 - 12 * p1 + 6 * p2),
            ]

            #Matrix forms
            #P(u)=U⋅M⋅G
            # Bézier : U=[u**3,u**2,u,1] 
            # Dérivative : U=[3*u**2,2*u,1,0] 
            self._M = np.array([ [-1,3,-3,1],[3,-6, 3, 0],[-3, 3, 0, 0],[ 1, 0, 0, 0]])
            self._G = np.array([p0, p1, p2, p3])


    def get_polynomial_coefficients(self):
        return self._C

    def get_derivative_polynomial_coefficients(self):
        return [self._D[0], self._D[1], self._D[2]]

    def evaluate(self, t):
        return self._C[0] * t**3 + self._C[1] * t**2 + self._C[2] * t + self._C[3]

    def evaluate_derivative(self, t):
        return self._D[0] * t**2 + self._D[1] * t + self._D[2]

    def evaluate_second_derivative(self, t):
        return self._S[0] * t + self._S[1]

    def derivative_roots(self):
        raise NotImplementedError("derivative_roots")

    def evaluate_curvature(self, t):
        B_prime = self.evaluate_derivative(t)
        B_double_prime = self.evaluate_second_derivative(t)
        return abs(
            B_prime.imag * B_double_prime.real - B_prime.real * B_double_prime.imag
        ) / (abs(B_prime) ** 3)

    def evaluate_multi(self, ts):
        U = np.array([[t**3, t**2, t, 1] for t in ts])        
        return U@self._M@self._G
    
    def evaluate_derivative_multi(self, ts):
        U = np.array([[3*t**2, 2*t, 1, 0] for t in ts])        
        return U@self._M@self._G

    def evaluate_second_derivative_multi(self, ts):
        U = np.array([[6*t, 2, 0, 0] for t in ts])        
        return U@self._M@self._G

    def evaluate_curvature_multi(self, ts):
        B_prime = self.evaluate_derivative_multi(ts)
        B_double_prime = self.evaluate_second_derivative_multi(ts)
        return abs(
            B_prime.imag * B_double_prime.real - B_prime.real * B_double_prime.imag
        ) / (abs(B_prime) ** 3)

    def to_svg(self) -> str:
        l = [f"""{z.real},{z.imag}""" for z in self._control_points]
        l.insert(0, "M")
        l.insert(2, "C")
        return " ".join(l)

    def x_aligned(self):
        
        o = self._control_points[0]
        p = self._control_points[-1]
        if abs(o.imag-p.imag)<1E-10:
            return CubicBezier( [ z-o for z in self._control_points] )
        bv = p - o
        rot = np.abs(bv)/bv
        return CubicBezier( [ ( (z-o)*rot ) for z in self._control_points] )


    def offseting(self, offset, N=3000):
        ε = .0000001
        _ts  = np.linspace(-1/N,1+1/N,N+2)
        ts = np.linspace(0,1,N)
        points = self.evaluate_multi(_ts)
        step = self.evaluate_derivative_multi(_ts) * np.exp(1j * np.pi / 2)
        step = step / np.abs(step)  
        newpoints = points+step*offset
        multicentres = center_multiple(newpoints)
        curvatures = np.abs(multicentres-newpoints[1:-1]) 


        newpoints = newpoints[1:-1]
        points = points[1:-1]
        
 
        if True:
            extremas = (argrelextrema(curvatures, np.greater)[0].tolist() + argrelextrema(curvatures, np.less)[0].tolist() )
            extremas.sort()
            if not extremas:
                extremas = [0,1]
            else:
                if extremas[0] != 0:
                    extremas.insert(0,0)
                n = len(curvatures)
                if extremas[-1] != n-1:
                    extremas.append(n-1)

        for i in extremas[:]:
            if i+1 in extremas: 
                extremas.remove(i+1)

        for i1, i2 in zip(extremas, extremas[1:]):

            if i2<extremas[-1]:
                i2 = i2+1
            apex, apexidx, c = _sub_offset_solve(self,ts, newpoints, i1, i2)
            
            if abs(apex-c.apex)>1 and i1<apexidx<i2:
                _,_, c = _sub_offset_solve(self,ts, newpoints, i1, apexidx+1)                
                yield c
                _,_, c = _sub_offset_solve(self,ts, newpoints, apexidx,i2)                
                yield c
            else:
                yield c



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


def main():


    
    styleorange = "fill: none; stroke: orange; stroke-opacity: 1; stroke-width: 1;"
    stylegreen = "fill: none; stroke: green; stroke-opacity: 1; stroke-width: 1;"
    stylegray = "fill: none; stroke: gray; stroke-opacity: 1; stroke-width: 1;"
    styleored = "fill: none; stroke: red; stroke-opacity: 1; stroke-width: 1;"
    styleblue = "fill: none; stroke: blue; stroke-opacity: 1; stroke-width: 1;"
    styleviolet = "fill: none; stroke: violet; stroke-opacity: 1; stroke-width: 1;"
    stylepurple3 = "fill: none; stroke: purple; stroke-opacity: 1; stroke-width: 3;"
    stylepurple = "fill: none; stroke: purple; stroke-opacity: 1; stroke-width: 1;"
    styleblack5 = "fill: none; stroke: black; stroke-opacity: 1; stroke-width: 5;"


    for n in range(5):
        print(n)
        file_path = f"svg\\output{n:05d}.svg"
        content = []
        angle = n/5*np.pi*2
        z1= 0-200j
        z2 = np.exp(1j*angle*4+.1)*300
        z3=  500+np.exp(.2+-1j*angle-.333)*300
        z4 = 500+200j

        #c = CubicBezier([ z + 500+1000j for z in [0j,100-500j, 400+600j, 500]])
        c = CubicBezier([ z + 500+1000j for z in [z1, z2, z3, z4]])
    

        for offset in   np.linspace(-1000,1000,100) :
            for i,co in enumerate( c.offseting(offset) ):
         
                p1, p2, p3, p4 = co._control_points                      
                d= f"M {p1.real},{p1.imag} C {p2.real},{p2.imag} {p3.real},{p3.imag} {p4.real},{p4.imag} "
                content.append(f"""<path d="{d}" style="{stylepurple}" data-num="{i}" />""")

        p1, p2, p3, p4 = c._control_points
        d= f"M {p1.real},{p1.imag} C {p2.real},{p2.imag} {p3.real},{p3.imag} {p4.real},{p4.imag} "
        content.append(f"""<path d="{d}" style="{styleblack5}" />""")
            
        content = "\n".join(content)
        content = f"""<svg width="1500" height="1500" xmlns="http://www.w3.org/2000/svg" style='background-color: white;'>
        <rect x="-10" y="-10" width="1520" height="1520" fill="white" stroke="white" />
        {content}

        </svg>"""

        with open(file_path, "w", encoding="utf-8") as fd:
            fd.write(content)






if __name__ == "__main__":
    main()