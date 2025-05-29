from svgpathtools import parse_path, CubicBezier, Line, Path
import cmath
import kdtree

import sample as SA1
import sample2 as SA2
import sample3 as SA3
import sample4 as SA4
import sample6 as SA6


# Class-based
class MagneticCanevas(object):

    def __init__(self, precision = 6):
        self.tree = kdtree.create(dimensions=2)
        self.prec = precision
        self.threshold = 10**(-precision)
        

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self.tree

    def get_point(self,p):
        if isinstance(p,complex):
            x,y = p.real, p.imag
        else:
            x,y = p
        v = (x,y)
        z = complex(x,y)
        s = self.tree.search_nn(v)
        if s:
            x,y = s[0].data
        if s and abs(complex(x,y)-z)<self.threshold:
            v =  s[0].data
        else:
            self.tree.add(v)
        return complex(round(v[0],self.prec+2), round(v[1],self.prec+2))
       

def spoil(canevas, c):
    if isinstance(c, CubicBezier):
        pts =  (c.start, c.control1, c.control2, c.end)
    if isinstance(c, Line):
        pts =  (c.start, c.end)
    return tuple(canevas.get_point(p) for p in pts)


def tosvg(c):
    if len(c)==4:
        return CubicBezier(*c)
    if len(c)==2:
        return Line(*c)

 

def get_curves(shape1, shape2):
    with MagneticCanevas(precision=3) as canevas:
        return [ spoil(canevas, c) for c in parse_path(shape1) ] , \
            [ spoil(canevas, c) for c in parse_path(shape2) ]  
    

def is_polygon_clockwise(vertices):
    """
    Determine if the polygon defined by the given vertices is clockwise.

    :param vertices: List of complex numbers representing the vertices of the polygon.
    :return: True if the polygon is clockwise, False otherwise.
    """
    sum = 0
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]
        sum += (v2.real - v1.real) * (v2.imag + v1.imag)
    
    return sum <= 0


POSITIVE = "pos"
NEGATIVE = "neg"
BOTH = "both"

def reindex_by_lowest(lst):
    if not lst:
        return lst  # Return empty list if input is empty
    
    min_index = lst.index(min(lst))  # Find the index of the lowest element
    return lst[min_index:] + lst[:min_index]  # Reindex the list

def dual(lst):
    return reindex_by_lowest( [-i for i in lst[::-1]])

def reverse(curves):
    return [ s[::-1] for s in curves[::-1]]

def combine(shape1, shape2, sens1=POSITIVE, sens2=POSITIVE, filename=None):
    curve1 ,curve2 = get_curves(shape1, shape2)
    curve1 = curve1
    if not filename:
        filename = f"output_{sens1}_{sens2}.svg"

 
    polygon = []
    for seg in curve1 :
        if polygon and polygon[-1] == seg[0]:
            polygon.extend(seg[1:])
        else:
            polygon.extend(seg)
    
    iscw = is_polygon_clockwise(polygon) 
    if not iscw:
        raise Exception("You should reverse curve 1", shape1)

    polygon = []
    for seg in curve2 :
        if polygon and polygon[-1] == seg[0]:
            polygon.extend(seg[1:])
        else:
            polygon.extend(seg)
    
    iscw = is_polygon_clockwise(polygon) 
    if not iscw:
        raise Exception("You should reverse curve 2", shape2)
    

    curves = [None]
    allowed = set()
    explains = {}

    
    for curve,sens, explain in ((curve1, sens1, "A"),(curve2, sens2, "B")):
        for c in curve:
            i = len(curves)
            curves.append(c)
            explains[i] = f"{explain}+"
            explains[-i] = f"{explain}-"
            if sens == POSITIVE:
                allowed.add(i)
            elif sens == NEGATIVE:
                allowed.add(-i)
            else :
                allowed.add(i)
                allowed.add(-i)


    isclosed1 = abs(curve1[-1][-1]-curve1[0][0])<.0001
    isclosed2 = abs(curve2[-1][-1]-curve2[0][0])<.0001
         

    print("allowed =",allowed)

        
    followings = {}
    for i,c in enumerate(curves):
        if i ==0:
            continue

        p = c[0]
        v = c[1]-c[0]
        
        if not p in followings:
            followings[p] = []
        followings[p].append((cmath.phase(v), i))
        
        c = c[::-1]
        i = -i
        p = c[0]
        v = c[1]-c[0]
        
        if not p in followings:
            followings[p] = []
        followings[p].append((cmath.phase(v), i))

    segments = []
    for k,v in followings.items():
        v = [ x[1] for x in sorted(v, key=lambda l: l[0]) ]
        for x1, x2 in zip(v, v[1:] + [v[0]]):
            segments.append([-x1,x2])

    for _ in range(len(segments)):
        for i1 in range(len(segments)):
            v1 = segments[i1]
            if v1 is not None: 
                for i2 in range(len(segments)):
                    v2 = segments[i2]
                    if i1!= i2 and v2 is not None:
                        if v1[-1] == v2[0] :
                            v1.extend(v2[1:])
                            segments[i1] = v1
                            segments[i2] = None
        if None in segments:
            segments = [x  for x in segments if x is not None]
        else:
            break



    #nettoyage
    segments2 = set()
 
    for s in segments:
        if  s[0] == s[-1] and len(s)>2:
            s = s[:-1]
        s = tuple(reindex_by_lowest(s))
        d =  tuple(reindex_by_lowest(dual(s)))
        if d in segments2:
            print("!!!!!!!!!! removing",s,d )
            segments2.remove(d)
        else:
            segments2.add(s)    

    segments = list(segments2)

    paths = []
    for segment in segments:

        explained =  ",".join([explains[i] for i in segment])
        print("*"*10,segment,explained)
        
        polygon = []
        for seg in [ curves[i] if i>=0 else tosvg(curves[-i][::-1]) for i in segment] :
            if polygon and polygon[-1] == seg[0]:
                polygon.extend(seg[1:])
            else:
                polygon.extend(seg)
        if not(sens1 == sens2 == POSITIVE):
            polygon=polygon[::-1]
        

        iscw = is_polygon_clockwise(polygon)
        testing = [ x for x in segment if not x in allowed]
        isok =  not any(testing)
        
            
        path = [ tosvg(curves[i]) if i>=0 else tosvg(curves[-i][::-1]) for i in segment]   
        

        d = Path(*path).d()

        isclosed = abs(path[0][0] - path[-1][-1])<.0001
        strokewidth = 1 if isclosed else 2
        if not isok :
            print("KO")
            #paths.append(f"""<path d="{d}" fill="yellow" stroke="orange" stroke-width="1" fill-opacity="0.5" "/>""")
        elif  iscw :
            print("CW")
            fill = "red" if isclosed else "none"
            paths.insert(0,f"""<path d="{d}" fill="{fill}" stroke="red" stroke-width="{strokewidth}" fill-opacity="0.5" "/>""")

        else:
            print("CCW")
            fill = "violet" if isclosed else "none"
            paths.append(f"""<path d="{d}" fill="{fill}" stroke="violet" stroke-width="{strokewidth}" fill-opacity="0.5" "/>""")

    strokewidth = 1 if isclosed1 else 2
    fill =  "grey" if isclosed1 else "none"
    paths.append(f"""<path d="{shape1}" id="A" fill="{fill}" stroke="green" stroke-width="{strokewidth}" style="opacity: 0.2;" sodipodi:insensitive="true" "/>""")
    
    strokewidth = 1 if isclosed1 else 2
    fill =  "grey" if isclosed2 else "none"
    paths.append(f"""<path d="{shape2}" id="B" fill="{fill}" stroke="blue" stroke-width="{strokewidth}" style="opacity: 0.2;" sodipodi:insensitive="true" "/>""")

 
    return "\n".join(paths)

    

if __name__ == "__main__":

    l = [ (POSITIVE, POSITIVE, "AUB +/+")
            , (POSITIVE, NEGATIVE, "B\A +/-")
            , (NEGATIVE, POSITIVE, "A\B -/+")
            , (NEGATIVE, NEGATIVE, "A∩B -/-")
            , (NEGATIVE, BOTH, "AxB -/±") 
            ]

    content = []
    si, sj = 0,0

    for a,b in [ (SA1.A,SA1.X),(SA2.S,SA2.W),(SA3.BODICE,SA3.CUT),(SA3.BODICE,SA3.CUT2)
                ,(SA3.BODICE,SA3.CUT3),(SA3.BODICE,SA3.CUT_closed),(SA4.S, SA4.O),(SA6.S, SA6.Z) ]:

        for (pos1, pos2, filename) in l:
                
            print("\n")
            print("="*100)
            print("-"*10, filename, "-"*10)
            paths =combine(a,b, pos1, pos2, filename)
            dx, dy = 50+(si)*300, 50+(sj)*300
            content.append(f"""<g transform="translate({dx},{dy})">
<text fill="black" font-family="Verdana" font-size="10">{filename}</text>
{paths}
</g>
    """)
            si = si+1
            if si>2:
                si, sj = 0,sj+1

        if si!=0 :
            si,sj = 0, sj+1









    content = "\n".join(content)
    content = f"""<svg width="1000" height="2000" xmlns="http://www.w3.org/2000/svg" style='background-color: white;'>
        {content}
    </svg>"""

    with open("output.svg", "w", encoding="utf-8") as fd:
        fd.write(content)

