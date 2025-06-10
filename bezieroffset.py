from typing import List
from bezier import bezier
import numpy as np
from scipy.ndimage import rotate
import cmath

content = []

def complex_to_bezier(control_points: List[complex]) -> bezier:

    # Convert complex numbers into a NumPy array (fortran order expected)
    nodes = np.asfortranarray([
        [point.real for point in control_points],
        [point.imag for point in control_points]
    ])

    # Create a Bézier curve
    return bezier.Curve(nodes, degree=len(control_points) - 1)

def bezier_to_svg(curve: bezier) -> str:
    l = [ f"{x},{y}" for x,y in curve.nodes.T ]
    l.insert(0,"M")
    l.insert(2,"C")
    return " ".join(l)


def bezier_to_complex(curve: bezier) -> List[complex]:
    return [ complex(x,y) for x,y in curve.nodes.T ]
    

def bezier_coefficients(P):
    """ Compute the coefficients for a cubic Bézier curve with complex control points. """
    C3 = P[3] - 3 * P[2] + 3 * P[1] - P[0]
    C2 = 3 * (P[2] - 2 * P[1] + P[0])
    C1 = 3 * (P[1] - P[0])
    C0 = P[0]
    return C3, C2, C1, C0

def find_inflection_points(P):
    """ Find the inflection points of a cubic Bézier curve with complex control points. """
    C3, C2, C1, _ = bezier_coefficients(P)

    # Quadratic equation coefficients using complex arithmetic
    a = 3 * (C2.real * C3.imag - C2.imag * C3.real)
    b = 3 * (C1.real * C3.imag - C1.imag * C3.real)
    c = C1.real * C2.imag - C1.imag * C2.real

    # Solve quadratic equation
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []  # No real solutions

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Filter valid t values in range [0,1]
    return [t for t in (t1, t2) if 0 <= t <= 1]

def min_max_of_bezier(pts: List[complex]) -> List[float]:
 
    a = np.angle( pts[-1] - pts[0])
    m = np.exp(-1j*a)
    ys = [ (m*z).imag for z in pts]
    zs = [ (z*m) for z in pts]
    
    if len(ys)==4:
        p0, p1, p2, p3 = ys
        #coefs = [p0, 3*(p1-p0), 3*(p0-2*p1+p2), p3-3*p2+3*p1-p0 ]
        # Compute derivative coefficients
        coefficients = [3 * (p1 - p0), 6 * (p2 - 2*p1 + p0), 3 * (p3 - 3*p2 + 3*p1 - p0)]
    

    # Find roots of the Bézier derivative
    roots = list(np.roots(coefficients[::-1]))  + find_inflection_points(pts)  
    #roots = find_inflection_points(pts)  
 
    roots.sort()

    print("roots", roots)
    d = bezier_to_svg(complex_to_bezier(zs))
    style = "fill: none; stroke: gray; stroke-opacity: 1; stroke-width: 1;"

    content.append(f"""<path d="{d}" style="{style}" />""")

    roots = [r for r in roots if 0<=r<=1] 
    if roots[0]>.2:
        roots.insert(0,0.0)
    else:
        roots[0] = 0.

    if roots[0]<.8:
        roots.append(1.0)
    else:
        roots[-1] = 1.

    return roots

def binomial_coeff(n, k):
    """Compute binomial coefficient manually."""
    if k > n:
        return 0
    result = 1
    for i in range(k):
        result *= (n - i) / (i + 1)
    return result

def bezier_derivative(t, control_points):
    """Compute the derivative of a Bézier curve."""
    n = len(control_points) - 1
    derivative_points = [n * (control_points[i+1] - control_points[i]) for i in range(n)]
    curve = 0.
    for i in range(n):
        curve += binomial_coeff(n-1, i) * ((1 - t) ** (n-1-i)) * (t ** i) * derivative_points[i]
    return curve

def bezier_curve(t, control_points):
    """Compute a Bézier curve."""
    n = len(control_points) - 1
    curve = 0.
    for i in range(n + 1):
        curve += binomial_coeff(n, i) * ((1 - t) ** (n - i)) * (t ** i) * control_points[i]
    return curve

def bezier_curvature_radius(t, control_points):
    """Calcul du rayon de courbure."""
    P_prime = bezier_derivative(t, control_points)
    P_double_prime = bezier_derivative(t, control_points[:-1])  # Approximation de la seconde dérivée
    curvature_radius = np.abs(P_prime)**3 / np.abs(P_prime * P_double_prime)
    return curvature_radius

def np_to_complex(a):
    [x],[y] = a.tolist()
    return complex(x,y)

def offset2(curve,d ):

    cp = bezier_to_complex(curve)

    s = 0.0
    p = np_to_complex( curve.evaluate(s) )
    t = np_to_complex( curve.evaluate_hodograph(s) )
    t = t / abs(t)
    n = t * cmath.exp(-1j * cmath.pi/2)
    r = bezier_curvature_radius(s, cp)
    c = p+n*r
    c0, r0, n0, t0 = (c,r,n,t)

    s = 1.0
    p = np_to_complex( curve.evaluate(s) )
    t = np_to_complex( curve.evaluate_hodograph(s) )
    t = t / abs(t)
    n = t * cmath.exp(-1j * cmath.pi/2)
    r = bezier_curvature_radius(s, cp)
    c = p+n*r 
    c1, r1, n1, t1 = (c,r,n,t)
 
 
    p0 = c0 -n0*(r0+d)
    p3 = c1 -n1*(r1+d)
    s = abs(p3-p0)/abs(cp[3]-cp[0])

    if s>1:
        s = s**.5    
    if s<1:
        s = s**1.2
 
 
    s0 = (r0+d)/r0*s  
    s1 = (r1+d)/r1*s  

 
    if d<0:     
        style = "fill: none; stroke: red; stroke-opacity: 1; stroke-width: 1;"
    else:
        style = "fill: none; stroke: green; stroke-opacity: 1; stroke-width: 1;"

    print("s0,s1",s0,s1)

    p0 = c0 -n0*(r0+d)
    p3 = c1 -n1*(r1+d)
    p1 = p0+(cp[1]-cp[0])*s0
    p2 = p3+(cp[2]-cp[3])*s1
    cp2 = [p0,p1,p2,p3]
    offsetcurve = complex_to_bezier(cp2)
    d = bezier_to_svg(offsetcurve)
    content.append(f"""<path d="{d}" style="{style}" />""")

    # content.append(f"""<circle cx="{p1.real}" cy="{p1.imag}" r="1" style="{style}" />""")
    # content.append(f"""<circle cx="{p2.real}" cy="{p2.imag}" r="1" style="{style}" />""")



def offset(curve,d ):
    cp = bezier_to_complex(curve)
    
    h = curve.evaluate_hodograph(0.0)
    [x],[y] = list(h)
    h = complex(x,y)
    h = h/abs(h)
    n = h * cmath.exp(-1j * cmath.pi/2)
    p = curve.evaluate(0.0)
    [x],[y] = list(p)
    p = complex(x,y)
    z0 = p + n *d

    h = curve.evaluate_hodograph(0.01)
    [x],[y] = list(h)
    h = complex(x,y)
    h = h/abs(h)
    n = h * cmath.exp(-1j * cmath.pi/2)
    p = curve.evaluate(0.01)
    [x],[y] = list(p)
    p = complex(x,y)
    z1 = p + n *d
    t0 = (z1-z0)/0.01





    h = curve.evaluate_hodograph(1.0)
    [x],[y] = list(h)
    h = complex(x,y)
    h = h/abs(h)
    n = h * cmath.exp(-1j * cmath.pi/2)
    p = curve.evaluate(1.0)
    [x],[y] = list(p)
    p = complex(x,y)
    z3 = p + n *d

    h = curve.evaluate_hodograph(0.99)
    [x],[y] = list(h)
    h = complex(x,y)
    h = h/abs(h)
    n = h * cmath.exp(-1j * cmath.pi/2)
    p = curve.evaluate(0.99)
    [x],[y] = list(p)
    p = complex(x,y)
    z2 = p + n *d
    t3 = (z2-z3)/0.01

    
    z1 = z0+t0/2
    z2 = z3+t3/2

    if d<0:     
        style = "fill: none; stroke: orange; stroke-opacity: 1; stroke-width: 1;"
    else:
        style = "fill: none; stroke: blue; stroke-opacity: 1; stroke-width: 1;"


    offsetcurve = complex_to_bezier([z0,z1,z2,z3])
    d = bezier_to_svg(offsetcurve)
    content.append(f"""<path d="{d}" style="{style}" />""")



control_points = [0, 100+120j, 400-180j, 500+0j ]
curve = complex_to_bezier(control_points)



# Create the Bézier curve
curve = complex_to_bezier(control_points)
ts = min_max_of_bezier(control_points)

style = "fill: none; stroke: black; stroke-opacity: 1; stroke-width: 1;"


for i,(t1, t2) in enumerate( zip(ts, ts[1:]) ):
    subcurve = curve.specialize(t1, t2)
    for d in [ -100,-75,-50,-25,25,50,75,100 ]:

 

        offset(subcurve, -d )





for t in ts:
    x,y = [ x[0] for x in  curve.evaluate(t).tolist()]
    content.append(f"""<circle cx="{x}" cy="{y}" r="1" style="{style}" />""")

style = "fill: violet; stroke: violet; stroke-opacity: 1; stroke-width: 1;"
for t in np.linspace(0,1.0,100):
    h = curve.evaluate_hodograph(t)
    [x],[y] = list(h)
    h = complex(x,y)
    h = h/abs(h)
    n = h * cmath.exp(-1j * cmath.pi/2)
    p = curve.evaluate(t)
    [x],[y] = list(p)
    p = complex(x,y)
    z = p + n *100
    content.append(f"""<circle cx="{z.real}" cy="{z.imag}" r="1" style="{style}" />""")

content = "\n".join(content)
content = f"""<svg width="500" height="500" xmlns="http://www.w3.org/2000/svg" style='background-color: white;'>
{content}
</svg>"""

with open(f"output.svg", "w", encoding="utf-8") as fd:
    fd.write(content)

