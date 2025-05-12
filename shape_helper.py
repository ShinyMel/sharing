from svgpathtools import svg2paths, CubicBezier, Path, Line, QuadraticBezier 
from pattern_crafting.point import Point
from pattern_crafting.shape import Shape
import bezier
import numpy as np

from collections import namedtuple

CurveSegment = namedtuple("CurveSegment", ["points", "slope", "curve", "index", "rindex"])



def get_intersections(curves1, curves2):
    """
    Computes the intersection points between curves of two shapes.

    Parameters:
    shape1 : object
        The first shape object containing its curves.
    shape2 : object
        The second shape object containing its curves.

    Returns:
    tuple
        A tuple containing two lists:
        - inters1: Intersection points of the curves in shape1 with curves in shape2.
        - inters2: Intersection points of the curves in shape2 with curves in shape1.

  
    """

    # Initialize empty lists to store intersection points for both shapes
    inters1 = []
    inters2 = []
    
    for i1, curve1 in enumerate(curves1):
        inters1.append([0.0, 1.0])  # Add default interval for each curve in shape1
        for i2, curve2 in enumerate(curves2):
            if i1 == 0:
                inters2.append([0.0, 1.0])  # Add default interval for each curve in shape2
            
            # Get intersection points between curve1 and curve2
            intersections = curve1.intersect(curve2)
            if intersections.size != 0:
                # Extend the interval lists with intersection points
                inters1[i1].extend(list(intersections[0, :]))
                inters2[i2].extend(list(intersections[1, :]))
    
    return inters1, inters2


def get_curve_segments(shape1, curves1, inters1, shape2, curves2, inters2):
    """
    Generates and returns specialized curve segments based on intersection points between two shapes.

    Parameters:
    shape1 : object
        The first shape object containing its curves.
    curves1 : list
        A list of curves that belong to shape1.
    inters1 : list
        Intersection points of the curves in shape1 with curves in shape2.
    shape2 : object
        The second shape object containing its curves.
    curves2 : list
        A list of curves that belong to shape2.
    inters2 : list
        Intersection points of the curves in shape2 with curves in shape1.

    Returns:
    list
        A list of dictionaries where each dictionary contains:
        - "c": A specialized curve segment.
        - "n1": The name or identifier of the starting point of the segment.
        - "n2": The name or identifier of the ending point of the segment.

     """
    

    n = 0
    curves = []
    
    for data in [zip(curves1, inters1), zip(curves2, inters2)]:
        for i1, (curve, ts) in enumerate(data):
            # Remove duplicate and sort intersection points
            ts = sorted(list(set(ts)))
            while ts[1] < 0.00001:
                ts.pop(1)

            while ts[-2] > 0.99999:
                ts.pop(-2)
            
            # Generate segment names based on intersection points
            names = [list(shape1.keys())[0]]
            l = len(ts)
            if l > 2:
                names.extend([f"_{n + i}_" for i in range(l - 2)])
            n = n + l - 2
            names.append(list(shape1.keys())[-1])
            
            # Create specialized curve segments based on intersection points
            for t1, t2, n1, n2 in zip(ts, ts[1:], names, names[1:]):
                curvepart = curve.specialize(t1, t2)
                p1 = Point(curvepart.nodes.T[0])
                p2 = Point(curvepart.nodes.T[-1])

                # Add curve segment to the list if it's not degenerate
                if p1.id != p2.id:
                   
                    #trunc points
                    nodes = curvepart.nodes
                    nodes[:,0] = p1.coo 
                    nodes[:,-1] = p2.coo 

                    curvepart=bezier.Curve( nodes , degree= curvepart.degree )

                    h = curvepart.evaluate_hodograph(0.)
                    s1 = complex(h[0], h[1])
                    if np.abs(s1)<.00001:
                        h = curvepart.evaluate_hodograph(.01)
                        s1 = complex(h[0], h[1])
                
                    h = curvepart.evaluate_hodograph(1.)
                    s2 = complex(h[0], h[1])
                    if np.abs(s2)<.00001:
                        h = curvepart.evaluate_hodograph(.99)
                        s2 = complex(h[0], h[1])


                    index = len(curves)
                    curves.append(CurveSegment(points=(p1.z, p2.z), slope=(s1, s2), curve=curvepart, index = index, rindex=index+1))
                    curves.append(CurveSegment(points=(p2.z, p1.z), slope=(-s2, -s1)
                                               , curve=bezier.Curve( nodes[:,::-1] , degree= curvepart.degree )
                                               , index = index+1, rindex=index))
                 
    return curves


def curvespart_to_svgpathtools(curve):
    if curve.degree == 1:
        # Extract control points from the bezier curve
        control_points = curve.nodes.T

        # Create a CubicBezier path segment
        return Line(
            start=complex(control_points[0, 0], control_points[0, 1]),
 
            end=complex(control_points[1, 0], control_points[1, 1])
        )
    elif curve.degree == 3:
        # Extract control points from the bezier curve
        control_points = curve.nodes.T

        # Create a CubicBezier path segment
        return CubicBezier(
            start=complex(control_points[0, 0], control_points[0, 1]),
            control1=complex(control_points[1, 0], control_points[1, 1]),
            control2=complex(control_points[2, 0], control_points[2, 1]),
            end=complex(control_points[3, 0], control_points[3, 1])
        )
    else:
        raise ValueError("TODO", curve.degree)

def extract_from_svg_file(svg_file):
 
    _, attributes = svg2paths(svg_file)
 
    paths = {}
    for   attribute in  attributes :
        s = Shape()
        d =  attribute["d"].strip()
        s.set_svg_path_d( d )
        paths[attribute["id"]] = s
    return paths


def curvepart_2_points(curve):
    arr = curve.nodes
    p2c = lambda p : complex(p[0], p[1])
    points = [Point(arr[:, i]) for i in range(arr.shape[1])]
    if len(points) > 2:
        points[1].handler = True
    if  len(points) > 3:
        points[2].handler = True
    return points



def cut(shape1, shape2, prefix="z"):
    """
    Computes the intersections between two shapes, generates specialized curve segments, 
    and processes them to find segment start and end points.

    Parameters:
    shape1 : object
        The first shape object containing its curves.
    shape2 : object
        The second shape object containing its curves.

    Returns:
    None

 

    """

    # Get Bezier package curves from shapes
    curves1 = list(shape1.get_bezierpkg_curves())
    curves2 = list(shape2.get_bezierpkg_curves())

  
    
    # Get intersections between curves of the two shapes
    inters1, inters2 = get_intersections(curves1, curves2)
    
    # Get curve segments based on intersection points
    curvesegments = get_curve_segments(shape1, curves1, inters1, shape2, curves2, inters2)

    nextsegment =  [ (c.rindex,100) for c in curvesegments ]


    for cs1 in curvesegments:
  
        for cs2 in curvesegments:
            if  cs1.points[1] == cs2.points[0] and cs1.index != cs2.index and cs1.rindex != cs2.index:            
                a = np.angle( cs2.slope[0]/cs1.slope[1]  ) 
                if a<nextsegment[cs1.index][1]:
                    nextsegment[cs1.index] = (cs2.index,a)
               
 

    edges = [ [i,x[0]] for i,x in  enumerate(nextsegment)]


    for _ in range(len(edges)):
        for i1 in range(len(edges)):
            e1 = edges[i1]
            for i2 in range(len(edges)):
                e2 = edges[i2]
                if i1!=i2 and e1 and e2 and e1[-1] == e2[0] :
                    e1.extend(e2[1:])
                    edges[i1] = e1
                    edges[i2] = None
        edges = [ x for x in edges if x]


    names1 = { p.id:n for n,p in shape1.items() }
    names2 = { p.id:n for n,p in shape2.items() }


    shapes = []
    for e in edges:
        if e:
            
            shaping = [ curvepart_2_points(curvesegments[i].curve) for i in e[:-1]  ]
            shape1 = Shape()
            shape2 = Shape()
            i1 = 0
            i2 = 0
            p_1 = "[ ]"
            for seg in shaping:
                for p in seg:
                    if p.id == p_1:
                        continue
                    p_1 = p.id
                    n1 = names1.get(p.id) or names2.get(p.id)
                    if not n1 or n1 in shape1:
                        i1 = i1+1
                        n1 = f"{prefix}{i1}"
                    p1 = p.copy()
                    p1.name = n1   
                    shape1[n1] = p1

                    n2 = names2.get(p.id) or names1.get(p.id)
                    if not n2 or n2 in shape2:
                        i2 = i2+1
                        n2 = f"{prefix}{i2}"
                    p2 = p.copy()
                    p2.name = n2   
                    shape2[n2] = p2
                
            shape = shape2 if i2<i1 else shape1
            shapes.append(shape)
    return shapes
