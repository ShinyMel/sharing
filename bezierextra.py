
import numpy as np
import cmath  


##################################################################################"""
##################################################################################"""
##################################################################################"""
##################################################################################"""
##################################################################################"""
# CF. https://stackoverflow.com/questions/35901079/calculating-the-inflection-point-of-a-cubic-bezier-curve
##################################################################################"""
##################################################################################"""
##################################################################################"""
##################################################################################"""
##################################################################################"""

import numpy as np

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



def bezier_coefficients(P):
    """ Compute the coefficients for a cubic Bézier curve. """
    C3 = P[3] - 3 * P[2] + 3 * P[1] - P[0]
    C2 = 3 * (P[2] - 2 * P[1] + P[0])
    C1 = 3 * (P[1] - P[0])
    return C3, C2, C1

def find_extrema(P):
    """ Find the extrema of a cubic Bézier curve. """
    C3, C2, C1 = bezier_coefficients(P)

    # Solve quadratic equation C3 * t^2 + C2 * t + C1 = 0
    a, b, c = 2 * C3, C2, C1
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return []  # No real solutions

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Filter valid t values in range [0,1]
    return [t for t in (t1, t2) if 0 <= t <= 1]

style = "fill: none; stroke: #000000; stroke-opacity: 1; stroke-width: 1;"





# Define control points
control_points =[0, 100+100j, 400-350j, 500 ]


d = [ f"{z.real},{z.imag}" for z in control_points ] 
d.insert(1,"C")
d.insert(0,"M")
d = " ".join(d)
content =  [f"""<path d="{d}" style="{style}" />"""] 
 
s = 0.0
p = bezier_curve(s, control_points)
t = bezier_derivative(s, control_points)
t = t / abs(t)
n = t * cmath.exp(-1j * cmath.pi/2)
r = bezier_curvature_radius(s, control_points)
c = p+n*r
#content.append(f"""<circle cx="{c.real}" cy="{c.imag}" r="{r}" style="{style}" />""")
c0, r0, n0, t0 = (c,r,n,t)

s = 1.0
p = bezier_curve(s, control_points)
t = bezier_derivative(s, control_points)
t = t / abs(t)
n = t * cmath.exp(-1j * cmath.pi/2)
r = bezier_curvature_radius(s, control_points)
c = p+n*r
#content.append(f"""<circle cx="{c.real}" cy="{c.imag}" r="{r}" style="{style}" />""")
c1, r1, n1, t1 = (c,r,n,t)

s =0

p = bezier_curve(s, control_points)
t = bezier_derivative(s, control_points)
t = t / abs(t)
n = t * cmath.exp(-1j * cmath.pi/2)
r = bezier_curvature_radius(s, control_points)
c = p+n*r
#content.append(f"""<circle cx="{c.real}" cy="{c.imag}" r="{r}" style="{style}" />""")
ch0, rh0, nh0, th0 =  (c,r,n,t)


s =.5

p = bezier_curve(s, control_points)
t = bezier_derivative(s, control_points)
t = t / abs(t)
n = t * cmath.exp(-1j * cmath.pi/2)
r = bezier_curvature_radius(s, control_points)
c = p+n*r
#content.append(f"""<circle cx="{c.real}" cy="{c.imag}" r="{r}" style="{style}" />""")
ch1, rh1, nh1, th1 = (c,r,n,t)


for d in [-150,-125,-100,-75,-50,-25,25,50,75,100,125,150]:
 
    p0 = c0 -n0*(r0+d)
    p3 = c1 -n1*(r1+d)
    s = abs(p3-p0)/abs(control_points[3]-control_points[0])
    if s>1:
        s = s**.5
    
    if s<1:
        s = s**1.2
 
 
    s0 = (r0+d)/r0 
    s1 = (r1+d)/r1  
    t = (s0+s1)/2
    s0 = (s0*.75+t*.25)*s
    s1 = (s1*.75+t*.25)*s
    s0 = t
 
 
    if d<0:
        #s0 = s0 *s
        #s1 = s1 *s
        style = "fill: none; stroke: red; stroke-opacity: 1; stroke-width: 1;"
    else:
        #s0 = s0 *s
        #s1 = s1 *s
        style = "fill: none; stroke: green; stroke-opacity: 1; stroke-width: 1;"

 

    p0 = c0 -n0*(r0+d)
    p3 = c1 -n1*(r1+d)
    p1 = p0+(control_points[1]-control_points[0])*s0
    p2 = p3+(control_points[2]-control_points[3])*s1
    control_points2 = [p0,p1,p2,p3]

    
    d = [ f"{z.real},{z.imag}" for z in control_points2 ] 
    d.insert(1,"C")
    d.insert(0,"M")
    d = " ".join(d)
    content.append(f"""<path d="{d}" style="{style}" />""")

# inflection_points = find_inflection_points(control_points)
# t = inflection_points[0]
# print("Inflection points at t =", t )
# z = bezier_curve(t , control_points)
# content.append(f"""<circle cx="{z.real}" cy="{z.imag}" r="5" style="{style}" />""")
ts = find_extrema([z.imag for z in control_points])
print("ts = ", ts)
for t in ts:
    z =  bezier_curve(t , control_points)
    content.append(f"""<circle cx="{z.real}" cy="{z.imag}" r="5" style="{style}" />""")


content = "\n".join(content)
content = f"""<svg width="500" height="500" xmlns="http://www.w3.org/2000/svg" style='background-color: white;'>
{content}
</svg>"""

with open(f"output.svg", "w", encoding="utf-8") as fd:
    fd.write(content)
