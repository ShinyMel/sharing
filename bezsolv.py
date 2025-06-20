from typing import List
from discrete2dcurve import Discrete2DCurve

from symbezier import SymBezier
import numpy as np

import warnings
warnings.filterwarnings('error')

def center(a, b, c):
    (x1, y1), (x2, y2), (x3, y3) = (a.real, a.imag), (b.real, b.imag), (c.real, c.imag)
    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    B = (
        (x1**2 + y1**2) * (y3 - y2)
        + (x2**2 + y2**2) * (y1 - y3)
        + (x3**2 + y3**2) * (y2 - y1)
    )
    C = (
        (x1**2 + y1**2) * (x2 - x3)
        + (x2**2 + y2**2) * (x3 - x1)
        + (x3**2 + y3**2) * (x1 - x2)
    )
    return complex(-B / A / 2, -C / A / 2)



def solve(p1, p4, dp1, dp4, apex):

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

    #print("ru = ", np.round(ru,3),"rv = ",np.round(rv,3),"rapex = ", np.round(rapex,3), "p1=",np.round(p1,3), "p4=",np.round(p4,3))
    

    t1 = (( 2 * ru.imag - rv.imag) + np.sqrt( ru.imag **2 -ru.imag * rv.imag + rv.imag**2)) / (3 * (ru.imag-rv.imag))
    t2 = (( 2 * ru.imag - rv.imag) - np.sqrt( ru.imag **2 -ru.imag * rv.imag + rv.imag**2)) / (3 * (ru.imag-rv.imag))
    t = [ t for t in (t1, t2) if 0<t<1][0]
    alpha = rapex.imag / (3*t*(ru.imag + t**2 * (ru.imag-rv.imag) + t * (-2*ru.imag + rv.imag)))
    p2 = q1 + u*alpha
    p3 = q4 + v*alpha
    return p2, p3


def offseting(sb: SymBezier, offset: float, content: List[str]):
    N = 500

    d = 1/N
    ts =[ -d] + list(np.linspace(0,1,N)) + [1+d, ] 
    pts = [ sb.evaluate(t) for t in ts ]
    referentials = [ sb.unit_referentiel(t) for t in ts ]
    
    zoff = [ z+n*offset for z,(_,n) in zip(pts, referentials)  ]

    curvatures = []
    for p1, p2, p3 in zip(zoff, zoff[1:], zoff[2:]):
        c = center(p1, p2, p3)
        r = abs(p2-c)
        curvatures.append(r)


    from scipy.signal import argrelextrema
    npcurvatures = np.array(curvatures)

    # Find indices of local maxima
    local_max_indices = argrelextrema(npcurvatures, np.greater)

    # Find indices of local minima
    local_min_indices = argrelextrema(npcurvatures, np.less)
    indices = (argrelextrema(npcurvatures, np.greater)[0].tolist() + argrelextrema(npcurvatures, np.less)[0].tolist() )
    indices.sort()
    if indices[0] != 0:
        indices.insert(0,0)
    n = len(curvatures)
    if indices[-1] != n-1:
        indices.append(n)
        
    print("extremas:", indices )
 
    d = [f"{z.real},{z.imag}" for z in sb._control_points]
    d.insert(0,"M")
    d.insert(2,"C")
    d = " ".join(d)
    style = "fill: none; stroke: gray; stroke-opacity: 1; stroke-width: 1;"
    #content.append(f"""<path d="{d}" style="{style}" />""")


    style = "fill: none; stroke: gray; stroke-opacity: 1; stroke-width: 1;"
    points = " ".join([ f"{p.real},{p.imag}"  for p in zoff])
    #content.append(f"""<polyline points="{points}" style="{style}" />""")


    style = "fill: none; stroke: red; stroke-opacity: 1; stroke-width: .5;"

    stop_ts = []
    for t, o, r1, r2, r3 in zip(ts[1:], zoff[1:], curvatures, curvatures[1:], curvatures[2:]):
        if (r2<r1 and r2<r3) or (r2>r1 and r2>r3):
            #content.append(f"""<circle cx="{o.real}" cy="{o.imag}" r="5" style="{style}" />""")
            stop_ts.append(t)

    #############################################################################################################

    stop_ts = [0.0] + [t for t in stop_ts if 0.01<t<.99 ] + [1.0]
    stop_ts.append((stop_ts[-1]+stop_ts[-2])/2)
    stop_ts.sort()

    
    for t1, t2 in zip(stop_ts, stop_ts[1:]):
        subts = [ t for t in ts if t1<=t<=t2 ]
        

        if len(subts)<2:
            print("TODO skipping")
            continue
        
        subzs = [ p for p,t in zip(zoff, ts) if t1<=t<=t2 ]
        if offset == 200:
            print(t1, t2, '=>', subts[0], subts[-1])


        curve = Discrete2DCurve(subzs)
        aligned = curve.x_aligned()
        i = np.argmax( [abs(v.imag) for v in aligned._values] )
        apex = curve._values[i]
        
        q1, q4 = subzs[0], subzs[-1]
        dq1 = sb.evaluate_derivative(t1)
        dq4 = -sb.evaluate_derivative(t2)
        dq1 = dq1/abs(dq1)
        dq4 = dq4/abs(dq4)

      
        
        style = "fill: violet; stroke: none; stroke-opacity: 1; stroke-width: 1;"
        #content.append(f"""<circle cx="{apex.real}" cy="{apex.imag}" r="5" style="{style}" />""")
        style = "fill: orange; stroke: none; stroke-opacity: 1; stroke-width: 1;"
        #content.append(f"""<circle cx="{q1.real}" cy="{q1.imag}" r="5" style="{style}" />""")
        
        q2, q3 = solve(q1, q4, dq1, dq4, apex)

        d = [f"{z.real},{z.imag}" for z in (q1, q2, q3, q4)]
        d.insert(0,"M")
        d.insert(2,"C")
        d = " ".join(d)
        style1 = "fill: none; stroke: black; stroke-opacity: 1; stroke-width: 2;"
        content.append(f"""<path d="{d}" style="{style1}" />""")

def main():
    content = []
    #sb = SymBezier([100+500j, 500+900j,  500+0j, 900+500j])
    sb = SymBezier([500+1000j, 1800+2000j,  0j, 1500+1000j])
    for d in range(50):
        offseting(sb, d*50 , content)

    content = "\n".join(content)
    content = f"""<svg width="2000" height="2000" xmlns="http://www.w3.org/2000/svg" style='background-color: white;'>
    {content}
    </svg>"""

    with open(f"output.svg", "w", encoding="utf-8") as fd:
        fd.write(content)



if __name__ == "__main__":
    main()
