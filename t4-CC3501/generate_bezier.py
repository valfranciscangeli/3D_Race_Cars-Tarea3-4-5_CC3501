""" Por Valeria Vallejos Franciscangeli
Basado en el Auxiliar 7, códigos de Pablo Pizarro"""

from curvas import *

show = False

#funcion que compara el punto actual con el punto previo guardado en recorrido
def punto_en_recorrido(punto, recorrido):
    if len(recorrido)!=0:
        punto_previo=recorrido[-1]
        if punto_previo[0] == punto[0] and punto_previo[1] == punto[1] and punto_previo[2] == punto[2]:
            return True
    return False


# funcion que genera los puntos de una curva de bezier.
def generate_bezier_curve(P0, P1, P2, P3):
    assert type(P0) == type(P1) == type(P2) == type(P3) == list

    # puntos de control:
    R0 = np.array([P0]).T
    R1 = np.array([P1]).T
    R2 = np.array([P2]).T
    R3 = np.array([P3]).T
    # generamos la curva
    GMb = bezierMatrix(R0, R1, R2, R3)
    bezierCurve = evalCurve(GMb, N=250)  # generamos N puntos por curva

    return bezierCurve


def generate_bezier_route():
    # Creamos 4 curvas que serán el recorrido del auto que se mueve automaticamente

    y = -0.037409  # altura del plano
    x_inicial = 2.25
    z_inicial = 5
    tension = 2.25
    #estirado = 1.5
    estirado= x_inicial

    # curva 1:
    R0 = [x_inicial, y, z_inicial]
    R1 = [x_inicial, y, z_inicial - 3]
    R2 = [x_inicial, y, z_inicial - 6]
    R3 = [x_inicial, y, z_inicial - 10]
    curva1 = generate_bezier_curve(R0, R1, R2, R3)

    # curva 2:
    R0 = [x_inicial, y, z_inicial - 10]
    R1 = [estirado, y, z_inicial - 10 - tension]
    R2 = [-estirado, y, z_inicial - 10 - tension]
    R3 = [-x_inicial, y, z_inicial - 10]
    curva2 = generate_bezier_curve(R0, R1, R2, R3)

    # curva 3:
    R0 = [-x_inicial, y, z_inicial - 10]
    R1 = [-x_inicial, y, z_inicial - 6]
    R2 = [-x_inicial, y, z_inicial - 3]
    R3 = [-x_inicial, y, z_inicial]
    curva3 = generate_bezier_curve(R0, R1, R2, R3)

    # curva 4:
    R0 = [-x_inicial, y, z_inicial]
    R1 = [-estirado, y, z_inicial + tension + 1.25]
    R2 = [estirado, y, z_inicial + tension + 1.25]
    R3 = [x_inicial, y, z_inicial]
    curva4 = generate_bezier_curve(R0, R1, R2, R3)

    return curva1, curva2, curva3, curva4


# funcion para graficar las curvas creadas
def show_bezier():
    curva1, curva2, curva3, curva4 = generate_bezier_route()
    #  Definimos la figura para 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plotCurve(ax, curva1, 'curva1', color=(1, 0, 0))
    plotCurve(ax, curva2, 'curva2', color=(1, 1, 0))
    plotCurve(ax, curva3, 'curva3', color=(0, 0, 1))
    plotCurve(ax, curva4, 'curva4', color=(0, 1, 0))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.savefig('ex_bezier.png')
    plt.show()


if show:
    show_bezier()


# funcion que une los cuatro segmentos para generar una unica curva
def get_route():
    recorrido = []
    curva1, curva2, curva3, curva4 = generate_bezier_route()
    curvas = [curva1, curva2, curva3, curva4]
    for i in range(4):
        curva = curvas[i]
        for punto in curva:
            if not(punto_en_recorrido(punto,recorrido)):
                recorrido.append(punto)
    return recorrido

if show:
    print(get_route())

# funcion que dibuja el recorrido creado para visualizar mejor si la forma está correcta
def dibuja_recorrido():
    import turtle
    recorrido = get_route()
    escala = 50
    for punto in recorrido:
        turtle.goto(punto[2] * escala, punto[0] * escala)

    turtle.done()


if show:
    dibuja_recorrido()
