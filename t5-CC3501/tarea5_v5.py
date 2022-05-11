"""
Quinta version de la Tarea 5
Realizada por Valeria Vallejos Franciscangeli
vvfranciscangeli@gmail.com

Basado en el código de Valentina Aguilar-Ivan Sipiran para la Tarea 4
Disenho de las AABB desde el código de Sebastian Olmos en el Auxiliar 11 - colisiones
Curso CC3501 de la FCFM
Semestre Primavera 2021

Comentado en español
Se dejan los comentarios del código base
Se añade implementacion de AABB's para detectar colisiones
En caso de colisionar auto a control con los objetos del mundo: no se puede mover
En caso de colisionar entre autos se quedan detenidos
"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from auxiliarT5 import *
from operator import add
import collisions as co
from dimensions import calculate_dimensions

__author__ = "Valeria Vallejos Franciscangeli-Valentina Aguilar-Ivan Sipiran"


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = False
        self.X = 2.0  # posicion X de donde está el auto
        self.Y = -0.037409  # posicion Y de donde está el auto
        self.Z = 6.5  # posicion Z de donde está el auto
        # lo siguiente se creó para poder usar coordenadas esfericas
        self.cameraPhiAngle = -np.pi / 4  # inclinacion de la camara
        self.cameraThetaAngle = np.pi / 2  # rotacion con respecto al eje y
        self.r = 2  # radio
        # agregamos un parametro para comenzar a chequear colisiones
        self.chek_collisions = True


# Esta clase contiene todos los parámetros de una luz Spotlight. Sirve principalmente para tener
# un orden sobre los atributos de las luces
class Spotlight:
    def __init__(self):
        self.ambient = np.array([0, 0, 0])
        self.diffuse = np.array([0, 0, 0])
        self.specular = np.array([0, 0, 0])
        self.constant = 0
        self.linear = 0
        self.quadratic = 0
        self.position = np.array([0, 0, 0])
        self.direction = np.array([0, 0, 0])
        self.cutOff = 0
        self.outerCutOff = 0


controller = Controller()

#  aquí se crea el pool de luces spotlight (como un diccionario)
spotlightsPool = dict()


# Esta función ejemplifica cómo podemos crear luces para nuestra escena. En este caso creamos 2 luces con diferentes
# parámetros

def setLights():
    # TAREA4: Primera luz spotlight
    spot1 = Spotlight()
    spot1.ambient = np.array([0.0, 0.0, 0.0])
    spot1.diffuse = np.array([1.0, 1.0, 1.0])
    spot1.specular = np.array([1.0, 1.0, 1.0])
    spot1.constant = 1.0
    spot1.linear = 0.09
    spot1.quadratic = 0.032
    spot1.position = np.array([2, 5, 0])  # TAREA4: esta ubicada en esta posición
    spot1.direction = np.array(
        [0, -1, 0])  # TAREA4: está apuntando perpendicularmente hacia el terreno (Y-, o sea hacia abajo)
    spot1.cutOff = np.cos(np.radians(12.5))  # TAREA4: corte del ángulo para la luz
    spot1.outerCutOff = np.cos(np.radians(45))  # TAREA4: la apertura permitida de la luz es de 45°
    # mientras más alto es este ángulo, más se difumina su efecto

    spotlightsPool['spot1'] = spot1  # TAREA4: almacenamos la luz en el diccionario, con una clave única

    # TAREA4: Segunda luz spotlight
    spot2 = Spotlight()
    spot2.ambient = np.array([0.0, 0.0, 0.0])
    spot2.diffuse = np.array([1.0, 1.0, 1.0])
    spot2.specular = np.array([1.0, 1.0, 1.0])
    spot2.constant = 1.0
    spot2.linear = 0.09
    spot2.quadratic = 0.032
    spot2.position = np.array([-2, 5, 0])  # TAREA4: Está ubicada en esta posición
    spot2.direction = np.array([0, -1, 0])  # TAREA4: también apunta hacia abajo
    spot2.cutOff = np.cos(np.radians(12.5))
    spot2.outerCutOff = np.cos(np.radians(15))  # TAREA4: Esta luz tiene menos apertura, por eso es más focalizada
    spotlightsPool['spot2'] = spot2  # TAREA4: almacenamos la luz en el diccionario

    # TAREA5: Luces spotlights para los faros de los autos
    spot3 = Spotlight()
    spot3.ambient = np.array([0, 0, 0])
    spot3.diffuse = np.array([1.0, 1.0, 1.0])
    spot3.specular = np.array([1.0, 1.0, 1.0])
    spot3.constant = 1.0
    spot3.linear = 0.09
    spot3.quadratic = 0.032
    spot3.position = np.array([2.10, 0.15, 4.8+1.5])  # posición inicial
    spot3.direction = np.array([0, -0.5, -1])  # dirección inicial
    spot3.cutOff = np.cos(np.radians(12.5))
    spot3.outerCutOff = np.cos(np.radians(30))
    spotlightsPool['spot3'] = spot3  # TAREA4: almacenamos la luz en el diccionario

    spot4 = Spotlight()
    spot4.ambient = np.array([0, 0, 0])
    spot4.diffuse = np.array([1.0, 1.0, 1.0])
    spot4.specular = np.array([1.0, 1.0, 1.0])
    spot4.constant = 1.0
    spot4.linear = 0.09
    spot4.quadratic = 0.032
    spot4.position = np.array([1.89, 0.15, 4.8+1.5])
    spot4.direction = np.array([0, -0.5, -1])
    spot4.cutOff = np.cos(np.radians(12.5))
    spot4.outerCutOff = np.cos(np.radians(30))
    spotlightsPool['spot4'] = spot4  # TAREA4: almacenamos la luz en el diccionario

    spot5 = Spotlight()
    spot5.ambient = np.array([0, 0, 0])
    spot5.diffuse = np.array([1.0, 1.0, 1.0])
    spot5.specular = np.array([1.0, 1.0, 1.0])
    spot5.constant = 1.0
    spot5.linear = 0.09
    spot5.quadratic = 0.032
    spot5.position = np.array([2.10, 0.15, 4.8])
    spot5.direction = np.array([0, -0.5, -1])
    spot5.cutOff = np.cos(np.radians(12.5))
    spot5.outerCutOff = np.cos(np.radians(30))
    spotlightsPool['spot5'] = spot5  # TAREA4: almacenamos la luz en el diccionario

    spot6 = Spotlight()
    spot6.ambient = np.array([0, 0, 0])
    spot6.diffuse = np.array([1.0, 1.0, 1.0])
    spot6.specular = np.array([1.0, 1.0, 1.0])
    spot6.constant = 1.0
    spot6.linear = 0.09
    spot6.quadratic = 0.032
    spot6.position = np.array([1.89, 0.15, 4.8])
    spot6.direction = np.array([0, -0.5, -1])
    spot6.cutOff = np.cos(np.radians(12.5))
    spot6.outerCutOff = np.cos(np.radians(30))
    spotlightsPool['spot6'] = spot6  # TAREA4: almacenamos la luz en el diccionario


# modificamos esta función para poder configurar todas las luces del pool
def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(60, float(width) / float(height), 0.1,
                                100)  # el primer parametro se cambia a 60 para que se vea más escena

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    # TAREA4: Como tenemos 2 shaders con múltiples luces, tenemos que enviar toda esa información a cada shader
    # TAREA4: Primero al shader de color
    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    # TAREA4: Enviamos la información de la luz puntual y del material
    # TAREA4: La luz puntual está desactivada por defecto (ya que su componente ambiente es 0.0, 0.0, 0.0),
    # pero pueden usarla para añadir más realismo a la escena
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].diffuse"), 0.0, 0.0, 0.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].specular"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].constant"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].linear"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].quadratic"), 0.01)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "pointLights[0].position"), 5, 5, 5)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.diffuse"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "material.specular"), 1.0, 1.0, 1.0)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "material.shininess"), 32)

    # TAREA4: Aprovechamos que las luces spotlight están almacenadas en el diccionario para mandarlas al shader
    for i, (k, v) in enumerate(spotlightsPool.items()):
        baseString = "spotLights[" + str(i) + "]."
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "ambient"), 1, v.ambient)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "diffuse"), 1, v.diffuse)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "specular"), 1, v.specular)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "constant"), v.constant)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "linear"), 0.09)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "quadratic"), 0.032)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "position"), 1, v.position)
        glUniform3fv(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "direction"), 1, v.direction)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "cutOff"), v.cutOff)
        glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, baseString + "outerCutOff"), v.outerCutOff)

    # TAREA4: Ahora repetimos todo el proceso para el shader de texturas con mútiples luces
    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].diffuse"), 0.0, 0.0, 0.0)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].specular"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].constant"), 0.1)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].linear"), 0.1)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].quadratic"), 0.01)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "pointLights[0].position"), 5, 5, 5)

    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.diffuse"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "material.specular"), 1.0, 1.0, 1.0)
    glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, "material.shininess"), 32)

    for i, (k, v) in enumerate(spotlightsPool.items()):
        baseString = "spotLights[" + str(i) + "]."
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "ambient"), 1, v.ambient)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "diffuse"), 1, v.diffuse)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "specular"), 1, v.specular)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "constant"), v.constant)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "linear"), 0.09)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "quadratic"), 0.032)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "position"), 1, v.position)
        glUniform3fv(glGetUniformLocation(texPipeline.shaderProgram, baseString + "direction"), 1, v.direction)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "cutOff"), v.cutOff)
        glUniform1f(glGetUniformLocation(texPipeline.shaderProgram, baseString + "outerCutOff"), v.outerCutOff)


# Esta función controla la cámara
def setView(texPipeline, axisPipeline, lightPipeline):
    # la idea de usar coordenadas esfericas para la camara fue extraida del auxiliar 6
    # como el auto reposa en el plano XZ, no sera necesaria la coordenada Y esferica.
    Xesf = controller.r * np.sin(controller.cameraPhiAngle) * np.cos(
        controller.cameraThetaAngle)  # coordenada X esferica
    Zesf = controller.r * np.sin(controller.cameraPhiAngle) * np.sin(
        controller.cameraThetaAngle)  # coordenada Y esferica

    viewPos = np.array([controller.X - Xesf, 0.5, controller.Z - Zesf])
    view = tr.lookAt(
        viewPos,  # eye
        np.array([controller.X, controller.Y, controller.Z]),  # at
        np.array([0, 1, 0])  # up
    )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(texPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    elif key == glfw.KEY_ENTER:
        controller.chek_collisions = not controller.chek_collisions

    # se agrega mensaje al utilizar teclas de movimiento
    elif key == glfw.KEY_A or key == glfw.KEY_LEFT:
        print('Left turn')

    elif key == glfw.KEY_D or key == glfw.KEY_RIGHT:
        print('Right turn')

    elif key == glfw.KEY_W or key == glfw.KEY_UP:
        print('Move Forward')

    elif key == glfw.KEY_S or key == glfw.KEY_DOWN:
        print('Move Backwards')

    elif key == glfw.KEY_V:
        print('SPEED CHANGE CONTROLADO')

    # elif key == glfw.KEY_B:
    #     print('SPEED CHANGE BEZIER')

    else:
        print('Unknown key')


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 4"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    # Se usan los shaders de múltiples luces
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = ls.MultipleLightTexturePhongShaderProgram()
    lightPipeline = ls.MultipleLightPhongShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    # Se cargan las texturas y se configuran las luces
    loadTextures()
    setLights()

    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)
    car1 = createCarScene(lightPipeline)  # Auto que seguirá la curva
    car_node = sg.findNode(car, 'car1')
    car_node.transform = tr.matmul([tr.translate(2.0, -0.037409, 6.5), tr.rotationY(np.pi)])

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    # ------------------AABBS para las colisiones------------------------------
    # haremos el analisis de colisiones utilizando AABB's ya que son mas adecuadas a las
    # geometrias de la escena

    # centro, dista x, dist y, dist z, escala

    world_boxes_list = co.AABBList(axisPipeline, [1, 0, 0])  # lista que almacena todas las AABB
    player_box_list = co.AABBList(axisPipeline, [1, 1, 0])  # lista que almacena todas las AABB
    bezier_box_list = co.AABBList(axisPipeline, [0, 1, 1])  # lista que almacena todas las AABB

    # --------------MUROS----------------

    # Se añade el AABB de muros
    for i in range(1, 5):
        nombre = 'wall' + str(i)
        posicion = sg.findPosition(dibujo, nombre).reshape(1, 4)[0][0:3]  # Se obtiene su posicion en el mundo
        world_boxes_list.objects += [co.AABB([posicion[0], posicion[1] + 0.2, posicion[2]], 0.1, 0.2,
                                             2.5)]  # Se crea el AABB con sus dimensiones y se añade a la lista

    # --------------CASAS-----------------

    # Se añade el AABB a cada casa
    for i in range(1, 11):
        nombre = "house" + str(i)
        posicion = sg.findPosition(dibujo, nombre).reshape(1, 4)[0][0:3]  # Se obtiene su posicion en el mundo
        world_boxes_list.objects += [co.AABB([posicion[0], posicion[1] + 0.4, posicion[2]], 0.5, 0.4,
                                             0.5)]  # Se crea el AABB con sus dimensiones y se añade a la lista

    # --------------AUTOS----------------
    posicion_inicial = np.array([2, 0.07, 5, 1])  # posicion inicial de las 2 AABB's
    posicion_inicial2 = np.array([2, 0.07, 6.5, 1])  # posicion inicial de las 2 AABB's
    dimensiones_init = np.array([0.14, 0.07, 0.31, 1])  # dimensiones iniciales de las 2 AABB's

    # puntos de las 4 esquinas de las AABB's al inicio

    p1_init = np.array([posicion_inicial[0] - dimensiones_init[0],
                        0.07,
                        posicion_inicial[2] - dimensiones_init[2], 1])

    p2_init = np.array([posicion_inicial[0] + dimensiones_init[0],
                        0.07,
                        posicion_inicial[2] - dimensiones_init[2], 1])

    p3_init = np.array([posicion_inicial[0] - dimensiones_init[0],
                        0.07,
                        posicion_inicial[2] + dimensiones_init[2], 1])

    p4_init = np.array([posicion_inicial[0] + dimensiones_init[0],
                        0.07,
                        posicion_inicial[2] + dimensiones_init[2], 1])

    p1_init2 = np.array([posicion_inicial2[0] - dimensiones_init[0],
                        0.07,
                        posicion_inicial2[2] - dimensiones_init[2], 1])

    p2_init2 = np.array([posicion_inicial2[0] + dimensiones_init[0],
                        0.07,
                        posicion_inicial2[2] - dimensiones_init[2], 1])

    p3_init2 = np.array([posicion_inicial2[0] - dimensiones_init[0],
                        0.07,
                        posicion_inicial2[2] + dimensiones_init[2], 1])

    p4_init2 = np.array([posicion_inicial2[0] + dimensiones_init[0],
                        0.07,
                        posicion_inicial2[2] + dimensiones_init[2], 1])

    # auto a control
    player_box_list.objects += [
        co.AABB(posicion_inicial2[0:3], dimensiones_init[0], dimensiones_init[1],
                dimensiones_init[2])]  # Se crea el AABB con sus dimensiones y se añade a la lista

    # auto automatico
    bezier_box_list.objects += [
        co.AABB(posicion_inicial[0:3], dimensiones_init[0], dimensiones_init[1],
                dimensiones_init[2])]  # Se crea el AABB con sus dimensiones y se añade a la lista

    # -------------------------------------------------------------------------

    # aplicación interactiva: velocidad del auto bezier
    def velocidad():
        v = (input("ingrese la velocidad deseada para el auto de la curva de Bezier: (puede ser: 1, 2 o 3) -->"))
        if v == "1":
            return 1000  # curva de 1000 puntos
        if v == "2":
            return 500  # curva de 500 puntos
        if v == "3":
            return 250  # curva de 250 puntos
        else:
            print("velocidad no permitida, se utilizará velocidad por defecto")
            return 1000  # curva de 1000 puntos


    #  Se genera la curva de la aplicación
    N = velocidad()  # menos puntos en la curva implica mayor velocidad
    C = generateCurveT5(N)

    step = 0

    # parametro iniciales
    t0 = glfw.get_time()
    coord_X = 0
    coord_Z = 0
    angulo = 0

    # definimos parametros candidatos iniciales (parametros del siguiente movimiento del auto controlado)
    # ----parametros iniciales
    coord_X_candidate = coord_X
    coord_Z_candidate = coord_Z
    angulo_candidate = angulo
    angle_candidate = 0
    # ----controlador
    controller_X_candidate = controller.X
    controller_Z_candidate = controller.Z
    controller_cameraThetaAngle_candidate = controller.cameraThetaAngle

    # Necesitamos los parámetros de posición y direcciones de las luces para manipularlas en el bucle principal
    light1pos = np.append(spotlightsPool['spot3'].position, 1)
    light2pos = np.append(spotlightsPool['spot4'].position, 1)
    dir_inicial = np.append(spotlightsPool['spot3'].direction, 1)

    light3pos = np.append(spotlightsPool['spot5'].position, 1)
    light4pos = np.append(spotlightsPool['spot6'].position, 1)
    dir_inicial2 = np.append(spotlightsPool['spot5'].direction, 1)

    # ---------------------------------------

    # aplicación interactiva: velocidad auto controlado
    def velocidad2():

        v = (input("ingrese la velocidad deseada para el auto a control: (puede ser: 1, 2 o 3) -->"))
        if v == '1':
            return 1
        if v == "2":
            return 2
        if v == "3":
            return 3
        else:
            print("velocidad no permitida, se utilizará velocidad por defecto")
            return 1


    velocidad = velocidad2()

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Se obtiene una diferencia de tiempo con respecto a la iteracion anterior.
        t1 = glfw.get_time()
        dt = (t1 - t0) * velocidad  # aumentar el delta implica mayor velocidad (angulo crece mas rapido)
        t0 = t1

        # ------------------------------------------------

        # se guardan valores de parametros candidatos del ciclo pasado

        coord_X_candidate_prev = coord_X_candidate
        coord_Z_candidate_prev = coord_Z_candidate
        angulo_candidate_prev = angulo_candidate
        controller_X_candidate_prev = controller_X_candidate
        controller_Z_candidate_prev = controller_Z_candidate
        controller_cameraThetaAngle_candidate_prev = controller_cameraThetaAngle_candidate
        angle_candidate_prev = angle_candidate

        # ------------------------------------------------

        # Se manejan las teclas de la animación

        # ir hacia adelante
        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            controller_X_candidate -= 1.5 * dt * np.sin(angulo_candidate)  # avanza la camara
            controller_Z_candidate -= 1.5 * dt * np.cos(angulo_candidate)  # avanza la camara
            coord_X_candidate -= 1.5 * dt * np.sin(angulo_candidate)  # avanza el auto
            coord_Z_candidate -= 1.5 * dt * np.cos(angulo_candidate)  # avanza el auto

        # ir hacia atras
        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            controller_X_candidate += 1.5 * dt * np.sin(angulo_candidate)  # retrocede la camara
            controller_Z_candidate += 1.5 * dt * np.cos(angulo_candidate)  # retrocede la cmara
            coord_X_candidate += 1.5 * dt * np.sin(angulo_candidate)  # retrocede el auto
            coord_Z_candidate += 1.5 * dt * np.cos(angulo_candidate)  # retrocede el auto

        # ir hacia la izquierda
        if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            controller_cameraThetaAngle_candidate -= dt  # camara se gira a la izquierda
            angulo_candidate += dt  # auto gira a la izquierda

        # ir hacia la derecha
        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            controller_cameraThetaAngle_candidate += dt  # camara se gira a la derecha
            angulo_candidate -= dt  # auto gira a la derecha

        # TECLAS DE OPCIONES-------------------------------------------------

        if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS:
            velocidad = velocidad2()

        # ---------------------------------------------------------------------

        # Boundig Box candidata del auto controlado, para revisar futuras colisiones

        # ------calculamos nuevos centros candidatos, se transforman al igual que el auto controlado

        # controlado
        boxc1_center_transform_candidate = tr.matmul([tr.translate(coord_X_candidate + 2, 0, coord_Z_candidate + 6.5),
                                                      tr.rotationY(np.pi + angulo_candidate),
                                                      tr.rotationY(-np.pi),
                                                      tr.translate(-2, 0, -6.5)])

        # calculamos nuevos posibles puntos de las esquinas de la AABB
        p1_new_c1_candidate = tr.matmul([boxc1_center_transform_candidate, p1_init2])
        p2_new_c1_candidate = tr.matmul([boxc1_center_transform_candidate, p2_init2])
        p3_new_c1_candidate = tr.matmul([boxc1_center_transform_candidate, p3_init2])
        p4_new_c1_candidate = tr.matmul([boxc1_center_transform_candidate, p4_init2])

        # calculamos nuevo posible centro
        nuevo_centro_c1_candidate = tr.matmul([boxc1_center_transform_candidate, posicion_inicial2])

        # calculamos nuevas posibles dimensiones
        dimensiones_nuevas_c1_candidate = calculate_dimensions(nuevo_centro_c1_candidate,
                                                               p1_new_c1_candidate,
                                                               p2_new_c1_candidate,
                                                               p3_new_c1_candidate,
                                                               p4_new_c1_candidate)

        # reemplazamos la AABB del player con los datos candidatos
        player_box_list.objects[0] = co.AABB(nuevo_centro_c1_candidate[0:3],
                                             dimensiones_nuevas_c1_candidate[0], 0.07,
                                             dimensiones_nuevas_c1_candidate[2])

        # ---------------------------------------------------------------------

        # chequeamos posibles colisiones
        if controller.chek_collisions:
            if not player_box_list.check_overlap(world_boxes_list) and not player_box_list.check_overlap(
                    bezier_box_list):
                # si no hay choque, actualizamos parametros de movimiento por los parametros candidatos
                coord_X = coord_X_candidate
                coord_Z = coord_Z_candidate
                angulo = angulo_candidate
                controller.X = controller_X_candidate
                controller.Z = controller_Z_candidate
                controller.cameraThetaAngle = controller_cameraThetaAngle_candidate

            else:
                print("estas chocando con algun objeto, intenta moverte en otra direccion")
                # si hay choque, volvemos los parametros candidatos a su valor del ciclo pasado
                coord_X_candidate = coord_X_candidate_prev
                coord_Z_candidate = coord_Z_candidate_prev
                angulo_candidate = angulo_candidate_prev
                controller_X_candidate = controller_X_candidate_prev
                controller_Z_candidate = controller_Z_candidate_prev
                controller_cameraThetaAngle_candidate = controller_cameraThetaAngle_candidate_prev
        else:
            coord_X = coord_X_candidate
            coord_Z = coord_Z_candidate
            angulo = angulo_candidate
            controller.X = controller_X_candidate
            controller.Z = controller_Z_candidate
            controller.cameraThetaAngle = controller_cameraThetaAngle_candidate

        # ---------------------------------------------------------------------

        # se actualiza angulo de movimiento del auto curva de Bezier
        if step < N * 4 - 1:
            angle_candidate = np.arctan2(C[step + 1, 0] - C[step, 0], C[step + 1, 2] - C[step, 2])
        else:
            angle_candidate = np.arctan2(C[0, 0] - C[step, 0], C[0, 2] - C[step, 2])

        # ---------------------------------------------------------------------

        # Boundig Box candidata del auto Bezier, para revisar futuras colisiones

        # ------calculamos nuevos centro candidatos, se transforman al igual que el auto Bezier

        # automatico (Bezier)

        boxc2_center_transform_candidate = tr.matmul([tr.translate(C[step, 0], 0, C[step, 2]),
                                                      tr.rotationY(angle_candidate),
                                                      tr.rotationY(-np.pi),
                                                      tr.translate(-2, 0, -5)])

        # calculamos nuevos puntos de las esquinas de la AABB
        p1_new_c2_candidate = tr.matmul([boxc2_center_transform_candidate, p1_init])
        p2_new_c2_candidate = tr.matmul([boxc2_center_transform_candidate, p2_init])
        p3_new_c2_candidate = tr.matmul([boxc2_center_transform_candidate, p3_init])
        p4_new_c2_candidate = tr.matmul([boxc2_center_transform_candidate, p4_init])

        # calculamos nuevo centro de la AABB
        nuevo_centro_c2_candidate = tr.matmul([boxc2_center_transform_candidate, posicion_inicial])

        # calculamos nuevas dimensiones de la AABB
        dimensiones_nuevas_c2_candidate = calculate_dimensions(nuevo_centro_c2_candidate,
                                                               p1_new_c2_candidate,
                                                               p2_new_c2_candidate,
                                                               p3_new_c2_candidate,
                                                               p4_new_c2_candidate)

        # actualizamos AABB del auto Bezier
        bezier_box_list.objects[0] = co.AABB(nuevo_centro_c2_candidate[0:3], dimensiones_nuevas_c2_candidate[0], 0.07,
                                             dimensiones_nuevas_c2_candidate[2])

        # ---------------------------------------------------------------------

        # chequeamos posibles colisiones

        if controller.chek_collisions:
            if not bezier_box_list.check_overlap(world_boxes_list) and not bezier_box_list.check_overlap(
                    player_box_list):
                # si no hay choque, actualizamos parametros de movimiento por los parametros candidatos
                angle = angle_candidate

            else:
                print("auto Bezier está chocando con algun objeto, intenta moverte en otra direccion")
                # si hay choque, volvemos los parametros candidatos a su valor del ciclo pasado
                angle = angle_candidate_prev
                step -= 1
        else:
            angle = angle_candidate

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if controller.fillPolygon:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        #  Se configura la cámara y el dibujo en cada iteración. Esto es porque necesitamos que en cada iteración
        # las luces de los faros de los carros se actualicen en posición y dirección
        setView(texPipeline, axisPipeline, lightPipeline)
        setPlot(texPipeline, axisPipeline, lightPipeline)

        # con CTRL IZQ se ven (o no) lineas de eje y AABB's
        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)
            # dibujamos AABB
            world_boxes_list.drawBoundingBoxes(axisPipeline)
            player_box_list.drawBoundingBoxes(axisPipeline)
            bezier_box_list.drawBoundingBoxes(axisPipeline)

        # se dibuja la escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")
        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model")
        sg.drawSceneGraphNode(car1, lightPipeline, "model")  # se agrega el nuevo auto

        # aqui se mueve el auto
        Auto = sg.findNode(car, 'system-car')
        Auto.transform = tr.matmul(
            [tr.translate(coord_X + 2, -0.037409, coord_Z + 6.5),
             tr.rotationY(np.pi + angulo),
             tr.rotationY(-np.pi),
             tr.translate(-2, 0.037409, -6.5)])
        # transformación que hace que el auto se ponga en el origen, para luego trasladarlo al
        # punto (2.0, −0.037409, 5.0) para después poder moverlo.

        # se configura su posición correspondiente
        carNode = sg.findNode(car1, "system-car")
        carNode.transform = tr.matmul(
            [tr.translate(C[step, 0], C[step, 1], C[step, 2]),
             tr.rotationY(angle),
             tr.rotationY(-np.pi),
             tr.translate(-2, 0.037409, -5)])
        # transformación que hace que el auto se ponga en el origen, para luego trasladarlo al
        # punto (2.0, −0.037409, 5.0) para después poder moverlo.

        # ---------------------------------------------------------------------------------------------------------

        # Las posiciones de las luces se transforman de la misma forma que el objeto
        posicion_transform = tr.matmul([tr.translate(coord_X + 2, -0.037409, coord_Z + 6.5),
                                        tr.rotationY(np.pi + angulo),
                                        tr.rotationY(-np.pi),
                                        tr.translate(-2, 0.037409, -6.5)])

        posicion3 = tr.matmul([posicion_transform, light1pos])
        posicion4 = tr.matmul([posicion_transform, light2pos])
        spotlightsPool['spot3'].position = posicion3
        spotlightsPool['spot4'].position = posicion4

        #  la dirección se rota con respecto a la rotación del objeto
        direccion = tr.matmul([tr.rotationY(angulo), dir_inicial])
        spotlightsPool['spot3'].direction = direccion
        spotlightsPool['spot4'].direction = direccion

        #  Hacemos lo mismo con las luces del segundo carro
        posicion_transform = tr.matmul([tr.translate(C[step, 0], C[step, 1], C[step, 2]),
                                        tr.rotationY(angle),
                                        tr.rotationY(-np.pi),
                                        tr.translate(-2, 0.037409, -5)])

        posicion5 = tr.matmul([posicion_transform, light3pos])
        posicion6 = tr.matmul([posicion_transform, light4pos])
        spotlightsPool['spot5'].position = posicion5
        spotlightsPool['spot6'].position = posicion6

        direccion = tr.matmul([tr.rotationY(np.pi + angle), dir_inicial2])
        spotlightsPool['spot5'].direction = direccion
        spotlightsPool['spot6'].direction = direccion

        # modificamos las posiciones de las AABB de los autos----------------------------------------------

        # ------calculamos nuevos centros, se transforman al igual que los autos

        # controlado
        boxc1_center_transform = tr.matmul([tr.translate(coord_X + 2, 0, coord_Z + 6.5),
                                            tr.rotationY(np.pi + angulo),
                                            tr.rotationY(-np.pi),
                                            tr.translate(-2, 0, -6.5)])

        # calculamos nuevos puntos de las esquinas de la AABB
        p1_new_c1 = tr.matmul([boxc1_center_transform, p1_init2])
        p2_new_c1 = tr.matmul([boxc1_center_transform, p2_init2])
        p3_new_c1 = tr.matmul([boxc1_center_transform, p3_init2])
        p4_new_c1 = tr.matmul([boxc1_center_transform, p4_init2])

        # calculamos nuevo centro de la AABB
        nuevo_centro_c1 = tr.matmul([boxc1_center_transform, posicion_inicial2])

        # calculamos nuevas dimensiones de la AABB
        dimensiones_nuevas_c1 = calculate_dimensions(nuevo_centro_c1, p1_new_c1, p2_new_c1, p3_new_c1, p4_new_c1)

        # actualizamos AABB del player
        player_box_list.objects[0] = co.AABB(nuevo_centro_c1[0:3], dimensiones_nuevas_c1[0], 0.07,
                                             dimensiones_nuevas_c1[2])

        # -----------------------------------------------------------

        # automatico (Bezier)

        boxc2_center_transform = tr.matmul([tr.translate(C[step, 0], 0, C[step, 2]),
                                            tr.rotationY(angle),
                                            tr.rotationY(-np.pi),
                                            tr.translate(-2, 0, -5)])

        # calculamos nuevos puntos de las esquinas de la AABB
        p1_new_c2 = tr.matmul([boxc2_center_transform, p1_init])
        p2_new_c2 = tr.matmul([boxc2_center_transform, p2_init])
        p3_new_c2 = tr.matmul([boxc2_center_transform, p3_init])
        p4_new_c2 = tr.matmul([boxc2_center_transform, p4_init])

        # calculamos nuevo centro de la AABB
        nuevo_centro_c2 = tr.matmul([boxc2_center_transform, posicion_inicial])

        # calculamos nuevas dimensiones de la AABB
        dimensiones_nuevas_c2 = calculate_dimensions(nuevo_centro_c2, p1_new_c2, p2_new_c2, p3_new_c2, p4_new_c2)

        # actualizamos AABB del auto Bezier
        bezier_box_list.objects[0] = co.AABB(nuevo_centro_c2[0:3], dimensiones_nuevas_c2[0], 0.07,
                                             dimensiones_nuevas_c2[2])

        # -------------------------------------------------------------------------------

        # Se realiza todo lo necesario para que car1 siga la curva
        step += 1
        if step > N * 4 - 1:
            step = 0

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()

    glfw.terminate()
