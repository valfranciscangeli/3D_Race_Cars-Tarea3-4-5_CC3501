
#funcion que calcula la maxima distancia (x, y, z) entre un centro y 4 puntos

def calculate_dimensions(centro, p1, p2, p3,p4):
    puntos=[p1, p2, p3,p4]
    maxX=0
    maxY=0
    maxZ=0
    for punto in puntos:
        x=abs(punto[0]-centro[0])
        y=abs(punto[1]-centro[1])
        z=abs(punto[2]-centro[2])
        if x>maxX:
            maxX=x
        if y>maxY:
            maxY=y
        if z>maxZ:
            maxZ=z
    return maxX, maxY, maxZ


