"""
Segunda version de la Tarea 4
Realizada por Valeria Vallejos Franciscangeli
vvfranciscangeli@gmail.com

Basado en el código de Valentina Aguilar-Ivan Sipiran para la Tarea 4
Curso CC3501 de la FCFM
Semestre Primavera 2021

Comentado en español
Se dejan los comentarios del código base
Se añade un auto que se mueve automáticamente por la pista derecha de la carretera.
Se añaden 4 luces tipo spotlight que simulan focos que se mueven con cada auto.
Se mantienen las luces tipo spotlight que se encuentran sobre las barreras de contencion, pero se modifica su apertura.

"""
import math
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from auxiliarT4 import *
from operator import add
import generate_bezier as gb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__author__ = "Valeria Vallejos Franciscangeli-Valentina Aguilar-Ivan Sipiran"


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.X = 2.0  # posicion X de donde esta el auto
        self.Y = -0.037409  # posicion Y de donde está el auto
        self.Z = 5.0  # posicion Z de donde esta el auto
        # lo siguiente se creó para poder usar coordenadas esfericas
        self.cameraPhiAngle = -np.pi / 4  # inclinacion de la camara
        self.cameraThetaAngle = np.pi / 2  # rotacion con respecto al eje y
        self.r = 2  # radio


# TAREA4: Esta clase contiene todos los parámetros de una luz Spotlight. Sirve principalmente para tener
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

# TAREA4: aquí se crea el pool de luces spotlight (como un diccionario)
spotlightsPool = dict()


# TAREA4: Esta función ejemplifica cómo podemos crear luces para nuestra escena.
# En este caso creamos 2 luces con diferentes parámetros

def setLights():
    # Primera luz spotlight
    spot1 = Spotlight()
    spot1.ambient = np.array([0.0, 0.0, 0.0])
    spot1.diffuse = np.array([1.0, 1.0, 1.0])
    spot1.specular = np.array([1.0, 1.0, 1.0])
    spot1.constant = 1.0
    spot1.linear = 0.09
    spot1.quadratic = 0.032
    spot1.position = np.array([2, 5, 0])  # se mantiene ubicacion original
    spot1.direction = np.array([0, -1, 0])  # se mantiene direccion original
    spot1.cutOff = np.cos(np.radians(10))  # corte del ángulo para la luz
    spot1.outerCutOff = np.cos(np.radians(15))  # cambiamos apertura maxima
    spotlightsPool['spot1'] = spot1  # TAREA4: almacenamos la luz en el diccionario, con una clave única

    # Segunda luz spotlight
    spot2 = Spotlight()
    spot2.ambient = np.array([0.0, 0.0, 0.0])
    spot2.diffuse = np.array([1.0, 1.0, 1.0])
    spot2.specular = np.array([1.0, 1.0, 1.0])
    spot2.constant = 1.0
    spot2.linear = 0.09
    spot2.quadratic = 0.032
    spot2.position = np.array([-2, 5, 0])  # se mantiene ubicacion original
    spot2.direction = np.array([0, -1, 0])  # se mantiene direccion original
    spot2.cutOff = np.cos(np.radians(10))
    spot2.outerCutOff = np.cos(np.radians(15))  # cambiamos apertura maxima
    spotlightsPool['spot2'] = spot2  # almacenamos la luz en el diccionario

    ######### Luces auto 1 (controlado) ########
    # foco DERECHO
    spot3 = Spotlight()
    spot3.ambient = np.array([0.0, 0.0, 0.0])
    spot3.diffuse = np.array([1.0, 1.0, 1.0])
    spot3.specular = np.array([1.0, 1.0, 1.0])
    spot3.constant = 1.0
    spot3.linear = 0.09
    spot3.quadratic = 0.032
    spot3.position = np.array([1.84, 0.15, 4.8])  # se calcula esta posicion segun la posicion inicial del auto
    spot3.direction = np.array([0, -0.5, -1])  # direccion diagonal hacia abajo
    spot3.cutOff = np.cos(np.radians(1))
    spot3.outerCutOff = np.cos(np.radians(30))
    spotlightsPool['spot3'] = spot3  # almacenamos la luz en el diccionario

    # foco IZQUIERDO
    spot4 = Spotlight()
    spot4.ambient = np.array([0.0, 0.0, 0.0])
    spot4.diffuse = np.array([1.0, 1.0, 1.0])
    spot4.specular = np.array([1.0, 1.0, 1.0])
    spot4.constant = 1.0
    spot4.linear = 0.09
    spot4.quadratic = 0.032
    spot4.position = np.array([1.64, 0.15, 4.8])  # se calcula esta posicion segun la posicion inicial del auto
    spot4.direction = spot3.direction  # direccion inicial igual a spotlight 3
    spot4.cutOff = spot3.cutOff  # le damos la misma apertura que la luz spotlight3
    spot4.outerCutOff = spot3.outerCutOff  # le damos el mismo angulo maximo que la luz spotlight3
    spotlightsPool['spot4'] = spot4  # almacenamos la luz en el diccionario

    ######### Luces auto 2 (automático) ########
    # usamos estos parametros para calcular la posicion inicial de las luces del segundo auto segun spotlight3
    distancia_autos = 0.5
    distancia_luces = 0.2
    # la direccion inicial de estas luces es la misma que para spotlight3

    # foco izquierdo
    spot5 = Spotlight()
    spot5.ambient = np.array([0.0, 0.0, 0.0])
    spot5.diffuse = np.array([1.0, 1.0, 1.0])
    spot5.specular = np.array([1.0, 1.0, 1.0])
    spot5.constant = 1.0
    spot5.linear = 0.09
    spot5.quadratic = 0.032
    spot5.position = np.add(spot4.position, np.array([distancia_autos, 0, 0]))
    spot5.direction = spot3.direction  # direccion inicial igual a spotlight 3
    spot5.cutOff = spot3.cutOff  # le damos la misma apertura que la luz spotlight3
    spot5.outerCutOff = spot3.outerCutOff  # le damos el mismo angulo maximo que la luz spotlight3
    spotlightsPool['spot5'] = spot5  # almacenamos la luz en el diccionario

    # foco derecho
    spot6 = Spotlight()
    spot6.ambient = np.array([0.0, 0.0, 0.0])
    spot6.diffuse = np.array([1.0, 1.0, 1.0])
    spot6.specular = np.array([1.0, 1.0, 1.0])
    spot6.constant = 1.0
    spot6.linear = 0.09
    spot6.quadratic = 0.032
    spot6.position = np.add(spot5.position, np.array([distancia_luces, 0, 0]))
    spot6.direction = spot5.direction  # direccion inicial igual a spotlight 3
    spot6.cutOff = spot3.cutOff  # le damos la misma apertura que la luz spotlight3
    spot6.outerCutOff = spot3.outerCutOff  # le damos el mismo angulo maximo que la luz spotlight3
    spotlightsPool['spot6'] = spot6  # almacenamos la luz en el diccionario


# TAREA4: modificamos esta función para poder configurar todas las luces del pool
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

    # TAREA4: Ahora repetimos todo el proceso para el shader de texturas con múltiples luces
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


# TAREA4: Esta función controla la cámara
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
        print('SPEED CHANGE ')

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
    # TAREA4: Se usan los shaders de múltiples luces
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

    # NOTA: Aqui creas un objeto con tu escena
    # TAREA4: Se cargan las texturas y se configuran las luces
    loadTextures()
    setLights()

    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)
    car2 = createCarScene(lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    # Parametros iniciales:---------------------------------------------------------------------

    # Camara y auto 1 -------
    t0 = glfw.get_time()
    coord_X = 0
    coord_Z = 0
    angulo = 0

    # Auto 2 -------
    i = 0
    base_transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0),
                                tr.rotationY(np.pi),
                                tr.rotationY(-np.pi),
                                tr.translate(-1.75, 0.037409, -5)])
    route = gb.get_route()
    angulo2 = 0

    # Luces-------
    spot3 = spotlightsPool['spot3']
    spot4 = spotlightsPool['spot4']
    spot5 = spotlightsPool['spot5']
    spot6 = spotlightsPool['spot6']
    sp3_pos_i = np.append(spot3.position, 1)
    sp4_pos_i = np.append(spot4.position, 1)
    sp5_pos_i = np.append(spot5.position, 1)
    sp6_pos_i = np.append(spot6.position, 1)
    dir_inicial = np.append(spot3.direction, 1)  # direccion inicial de las luces

    # Velocidad -------

    def preguntar_velocidad():
        velocidad = float(input('velocidad? (elegir un numero positivo, se recomienda 2.0)'))
        if velocidad <= 0:
            input('valor erroneo, por favor oprima ENTER y vuelva a ingresar una velocidad')
            return preguntar_velocidad()
        else:
            return velocidad


    velocidad = preguntar_velocidad()

    # INICIO BUCLE:-------------------------------------------------------------------------------------------

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Se obtiene una diferencia de tiempo con respecto a la iteracion anterior.
        t1 = glfw.get_time()
        dt = t1 - t0
        dt *= velocidad  # aqui se maneja la velocidad
        t0 = t1

        # TECLAS DE ANIMACION-------------------------------------------------
        # ir hacia adelante
        if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            controller.X -= 1.5 * dt * np.sin(angulo)  # avanza la camara
            controller.Z -= 1.5 * dt * np.cos(angulo)  # avanza la camara
            coord_X -= 1.5 * dt * np.sin(angulo)  # avanza el auto
            coord_Z -= 1.5 * dt * np.cos(angulo)  # avanza el auto

        # ir hacia atras
        if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            controller.X += 1.5 * dt * np.sin(angulo)  # retrocede la camara
            controller.Z += 1.5 * dt * np.cos(angulo)  # retrocede la cmara
            coord_X += 1.5 * dt * np.sin(angulo)  # retrocede el auto
            coord_Z += 1.5 * dt * np.cos(angulo)  # retrocede el auto

        # ir hacia la izquierda
        if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            controller.cameraThetaAngle -= dt  # camara se gira a la izquierda
            angulo += dt  # auto gira a la izquierda

        # ir hacia la derecha
        if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS) or (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            controller.cameraThetaAngle += dt  # camara se gira a la derecha
            angulo -= dt  # auto gira a la derecha

        # TECLAS DE OPCIONES-------------------------------------------------

        if glfw.get_key(window, glfw.KEY_V) == glfw.PRESS:
            velocidad = preguntar_velocidad()

        # ------------------------------------------------------------------

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if controller.fillPolygon:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # CONFIGURACION DE CAMARA Y DIBUJO:---------------------------------------------------------

        setView(texPipeline, axisPipeline, lightPipeline)
        setPlot(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        # DIBUJO DE ESCENA:--------------------------------------------------------------------------
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")

        glUseProgram(lightPipeline.shaderProgram)

        # ANIMACION AUTO 1:--------------------------------------------------------------------------
        sg.drawSceneGraphNode(car, lightPipeline, "model")
        Auto = sg.findNode(car, 'system-car')
        Auto.transform = tr.matmul([tr.translate(coord_X + 2, -0.037409, coord_Z + 5),
                                    tr.rotationY(np.pi + angulo),
                                    tr.rotationY(-np.pi),
                                    tr.translate(-2.25, 0.037409, -5)])
        # transformación que hace que el auto se ponga en el origen,
        # para luego trasladarlo al punto (2.0, −0.037409, 5.0) para despés poder moverlo.

        # ANIMACION AUTO 2:--------------------------------------------------------------------------
        sg.drawSceneGraphNode(car2, lightPipeline, "model")
        Auto2 = sg.findNode(car2, 'system-car')
        positionAuto = sg.findPosition(car2, 'car1')  # posición actual completa auto
        position = [positionAuto[0, 0], positionAuto[1, 0], positionAuto[2, 0]]  # x, y y z de posición del auto
        next_point = route[i]  # siguiente coordenada segun curva Bezier

        # Se modifica angulo de rotacion.
        largo_ruta = len(route)
        largo_parte_ruta = largo_ruta / 4  # largo de 1 segmento (4) de la curva completa Bezier

        # se incrementa angulo solo cuando el auto se encuentra en la parte no recta de la curva
        if largo_parte_ruta <= i <= largo_parte_ruta * 2 or largo_parte_ruta * 3 <= i <= largo_ruta:
            angle = np.pi / largo_parte_ruta  # 180°/numero de puntos segmento ruta
            angulo2 += np.pi / largo_parte_ruta  # aqui se guarda el cambio de angulo acumulado
        else:
            angle = 0

        # Se modifica la tranformacion del auto 2 para realizar la animacion
        # se calcula segun la posicion previa del auto y la posicion siguiente segun la ruta Bezier
        translation = tr.translate(next_point[0] - position[0],
                                   next_point[1] - position[1],
                                   next_point[2] - position[2])
        new_transform = tr.matmul([tr.rotationY(angle),
                                   translation])

        # new_transform = tr.identity()  # descomentar para poder ver el auto en posicion inicial
        Auto2.transform = tr.matmul([new_transform, base_transform])  # redefinimos la transformacion del nodo Auto2

        # Seteamos tranformacion base como la transformacion recien realizada
        base_transform = tr.matmul([new_transform, base_transform])

        # LUCES:-----------------------------------------------------------------------------------

        # AUTO 1------------------------------

        # Cambio de posicion:------------

        # Se calcula con la misma idea que la animacion del auto 1
        posicion_transform = tr.matmul([tr.translate(coord_X + 2, -0.037409, coord_Z + 5),
                                        tr.rotationY(np.pi + angulo),
                                        tr.rotationY(-np.pi),
                                        tr.translate(-2, 0.037409, -5)])

        posicion3 = tr.matmul([posicion_transform, sp3_pos_i])
        posicion4 = tr.matmul([posicion_transform, sp4_pos_i])
        # cambiamos el atributo posicion de las spotlights 4 y 5
        spot3.position = posicion3
        spot4.position = posicion4

        # Cambio de direccion:------------

        # Se calcula rotando la direccion inicial segun el mismo angulo de rotacion del auto 1

        direccion = tr.matmul([tr.rotationY(angulo), dir_inicial])
        # cambiamos el atributo direccionde las spotlights 3 y 4
        spot3.direction = direccion
        spot4.direction = direccion

        # AUTO 2------------------------------

        # Cambio de posicion:------------

        # Se calcula con la misma idea que la animacion del auto 2

        posicion5 = tr.matmul([tr.rotationY(angle),
                               translation, sp5_pos_i])
        sp5_pos_i = posicion5  # seteamos la nueva direccion como direccion inicial
        posicion6 = tr.matmul([tr.rotationY(angle),
                               translation, sp6_pos_i])
        sp6_pos_i = posicion6  # seteamos la nueva direccion como direccion inicial

        # cambiamos el atributo posicion de las spotlights 5 y 6
        spot5.position = posicion5
        spot6.position = posicion6

        # Cambio de direccion:------------

        # Se calcula rotando la direccion inicial segun el angulo de rotacion acumulado del auto 2

        direccion = tr.matmul([tr.rotationY(angulo2), dir_inicial])
        # cambiamos el atributo direccion de las spotlights 5 y 6
        spot5.direction = direccion
        spot6.direction = direccion

        # CONFIGURACION BUCLE:--------------------------------------------------------------------------

        # Incrementamos indice
        i += 1
        # Si quedamos fuera de index, se resetea i
        if i >= len(route):
            i = 0

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

        # FIN BUCLE:------------------------------------------------------------------------------------------------

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()

    glfw.terminate()
