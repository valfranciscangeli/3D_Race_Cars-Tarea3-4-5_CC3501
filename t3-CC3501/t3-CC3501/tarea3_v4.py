"""
Cuarta version de la tarea 3
Realizada por Valeria Vallejos Franciscangeli
vvfranciscangeli@gmail.com

Basado en el código de Ivan Sipiran para la Tarea 3
Curso CC3501 de la FCFM
Semestre Primavera 2021

Comentado en español
Se dejan los comentarios del código base
Se añaden explicaciones sobre creación de la escena estática,
fondo de ventana y movimientos debidos al controlador

"""

__author__ = "Valeria Vallejos Franciscangeli"
__license__ = "MIT"


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
from operator import add


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        # self.viewPos = np.array([12,12,12])        # eye original del código
        self.viewPos = np.array([2.0, 1, 7.0])  # eye detrás del auto en su posición original
        # self.at = np.array([0,0,0])                # at original del código
        self.at = np.array([2.0, -0.037409, 5.0])  # at detrás del auto en su posición original
        self.camUp = np.array([0, 1, 0])
        self.distance = 20


controller = Controller()


def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width) / float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)

    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)


def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
        controller.viewPos,
        controller.at,
        controller.camUp
    )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"),
                controller.viewPos[0],
                controller.viewPos[1],
                controller.viewPos[2])


# se dejan los valores originales de cámara del código al presionar las teclas numéricas
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

    elif key == glfw.KEY_1:
        controller.viewPos = np.array([controller.distance,
                                       controller.distance,
                                       controller.distance])  # Vista diagonal 1
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código


    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0, 0, controller.distance])  # Vista frontal
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance, 0, controller.distance])  # Vista lateral
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0, controller.distance, 0])  # Vista superior
        controller.camUp = np.array([1, 0, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_5:
        controller.viewPos = np.array(
            [controller.distance, controller.distance, -controller.distance])  # Vista diagonal 2
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_6:
        controller.viewPos = np.array(
            [-controller.distance, controller.distance, -controller.distance])  # Vista diagonal 3
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_7:
        controller.viewPos = np.array(
            [-controller.distance, controller.distance, controller.distance])  # Vista diagonal 4
        controller.camUp = np.array([0, 1, 0])
        controller.at = np.array([0, 0, 0])  # at original del código

    elif key == glfw.KEY_A or key == glfw.KEY_LEFT:
        print('Left turn')

    elif key == glfw.KEY_D or key == glfw.KEY_RIGHT:
        print('Right turn')

    elif key == glfw.KEY_W or key == glfw.KEY_UP:
        print('Forward')

    elif key == glfw.KEY_S or key == glfw.KEY_DOWN:
        print('Backward')
    else:
        print('Unknown key')


def createOFFShape(pipeline, filename, r, g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape


def readOFF(filename, color):
    vertices = []
    normals = []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line == "OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]

        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices, 3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]

            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1],
                    vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1],
                    vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]
            normals[aux[1]][1] += res[1]
            normals[aux[1]][2] += res[2]

            normals[aux[2]][0] += res[0]
            normals[aux[2]][1] += res[1]
            normals[aux[2]][2] += res[2]

            normals[aux[3]][0] += res[0]
            normals[aux[3]][1] += res[1]
            normals[aux[3]][2] += res[2]
            # print(faces)
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0], :]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1], :]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2], :]
            vertexDataF += vertex.tolist()

            indices += [index, index + 1, index + 2]
            index += 3

        return bs.Shape(vertexDataF, indices)


def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape


def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d + 1.0, 0.0, 0.0, 1.0, 0.0]

    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d], [0], [0], [1]]))
        p2 = rot.dot(np.array([[d + 1], [0], [0], [1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)

        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont / 4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont / 4])

        indices.extend([currentIndex1, currentIndex2, currentIndex2 + 1])
        indices.extend([currentIndex2 + 1, currentIndex2 + 2, currentIndex1])

        if cont > 4:
            cont = 0

        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont / 4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont / 4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)


def createTiledFloor(dim):
    vert = np.array([[-0.5, 0.5, 0.5, -0.5], [-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
                    np.float32)
    rot = tr.rotationX(-np.pi / 2)
    vert = rot.dot(vert)

    indices = [
        0, 1, 2,
        2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim, dim, 1):
        for j in range(-dim, dim, 1):
            tra = tr.translate(i, 0.0, j)
            newVert = tra.dot(vert)

            v = newVert[:, 0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:, 1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:, 2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:, 3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])

            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)


# función createHouse que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)
def createHouse(pipeline, archivoTexturaPared, archivoTexturaTecho, archivoTexturaPuerta, archivoTexturaVentana):
    # PARÁMETROS A USAR: ----------------------

    escalaTecho = 0.7
    escalaPared = 1
    escalaLadoPared = 1

    # PRIMITIVAS:-------------------------------

    techo = createGPUShape(pipeline, bs.createTextureQuad(escalaTecho, escalaPared))
    techo.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaTecho), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    frenteTecho = createGPUShape(pipeline, bs.createTextureTriangle(escalaTecho, escalaPared))
    frenteTecho.texture = techo.texture

    pared = createGPUShape(pipeline, bs.createTextureQuad(escalaPared, escalaPared))
    pared.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaPared), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    paredLado = createGPUShape(pipeline, bs.createTextureQuad(escalaTecho, escalaLadoPared))
    paredLado.texture = pared.texture

    ventana = createGPUShape(pipeline, bs.createTextureQuad(1, 1))
    ventana.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaVentana), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    puerta = createGPUShape(pipeline, bs.createTextureQuad(1, 1))
    puerta.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaPuerta), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    # getAssetPath('door3.jpg'), GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    # NODOS:--------------------------------

    # Lados de la casa:-------------------
    nodoTecho = sg.SceneGraphNode('techo')
    nodoTecho.transform = tr.matmul([tr.translate(0.1, 0.025, 0), tr.rotationZ(np.pi / 5),
                                     tr.translate(0.2, 0.8, -0.35),
                                     tr.rotationY(np.pi / 2),
                                     tr.scale(escalaTecho, escalaPared, 1)])
    nodoTecho.childs += [techo]

    nodoParedLado = sg.SceneGraphNode('pared')
    nodoParedLado.transform = tr.matmul([tr.translate(0, 0, -0.35),
                                         tr.rotationY(np.pi / 2),
                                         tr.scale(escalaTecho, escalaLadoPared, 1)])
    nodoParedLado.childs += [pared]

    nodoLadoCasa = sg.SceneGraphNode('lado_casa')
    nodoLadoCasa.transform = tr.translate(0.5, 0, 0)
    nodoLadoCasa.childs += [nodoTecho, nodoParedLado]

    nodoOtroLadoCasa = sg.SceneGraphNode('otro_lado_casa')
    nodoOtroLadoCasa.transform = np.array([
        # reflejamos en x (lo vimos en aux 4)
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    nodoOtroLadoCasa.childs += [nodoLadoCasa]

    # Frente de la casa:--------------------
    nodoFrenteTecho = sg.SceneGraphNode('frente_techo')
    nodoFrenteTecho.transform = tr.matmul([tr.translate(0, 0.85, 0),
                                           tr.scale(1, escalaTecho, 1)])
    nodoFrenteTecho.childs += [frenteTecho]

    nodoPared = sg.SceneGraphNode('pared')
    nodoPared.transform = tr.scale(1, escalaPared, 1)
    nodoPared.childs += [pared]

    nodoFrenteCasa = sg.SceneGraphNode('frente')
    nodoFrenteCasa.transform = tr.identity()
    nodoFrenteCasa.childs += [nodoPared,
                              nodoFrenteTecho]

    # Atrás de la casa:-------------------
    nodoAtrasCasa = sg.SceneGraphNode('atras')
    nodoAtrasCasa.transform = tr.translate(0, 0, -0.7)
    nodoAtrasCasa.childs += [nodoFrenteCasa]

    # Decoraciones:-------------------------
    nodoPuerta = sg.SceneGraphNode('puerta')
    nodoPuerta.transform = tr.matmul([tr.translate(0, -0.25, 0.001),
                                      tr.scale(0.3, 0.5, 1)])
    nodoPuerta.childs += [puerta]

    nodoVentana1 = sg.SceneGraphNode('ventana_1')
    nodoVentana1.transform = tr.matmul([tr.translate(0.3, 0.2, 0.001),
                                        tr.scale(0.3, 0.3, 1)])
    nodoVentana1.childs += [ventana]

    nodoVentana2 = sg.SceneGraphNode('ventana_2')
    nodoVentana2.transform = np.array([
        # reflejamos en x (lo vimos en aux 4)
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    nodoVentana2.childs += [nodoVentana1]

    # Casa completa:-------------------------
    nodoCasa = sg.SceneGraphNode('casa')
    nodoCasa.transform = tr.translate(0, 0.5 - (1 - escalaPared) - 0.037409, 0)
    nodoCasa.childs += [nodoOtroLadoCasa,
                        nodoLadoCasa,
                        nodoFrenteCasa,
                        nodoAtrasCasa,
                        nodoPuerta,
                        nodoVentana1,
                        nodoVentana2
                        ]

    return nodoCasa


# función createWall que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)
def createWall(pipeline, archivoTexturaMuro):
    # PARÁMETROS A USAR:----------------------------------------
    escalaMuro = 0.3

    # PRIMITIVAS:-----------------------------------------------
    muro = createGPUShape(pipeline, bs.createTextureQuad(1.0, escalaMuro))
    muro.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaMuro), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    lado = createGPUShape(pipeline, bs.createTextureQuad(escalaMuro, escalaMuro))
    lado.texture = es.textureSimpleSetup(
        getAssetPath(archivoTexturaMuro), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    # NODOS:---------------------------------------------------------
    # Cara frontal:----------------
    muro1 = sg.SceneGraphNode('muro_1')
    muro1.transform = tr.matmul([tr.scale(1, escalaMuro, 1),
                                 tr.translate(0, 0.5 - escalaMuro - 0.037409, 0)])
    muro1.childs += [muro]

    # Cara trasero:-------------------
    muro2 = sg.SceneGraphNode('muro_2')
    muro2.transform = tr.translate(0, 0, -0.3)
    muro2.childs += [muro1]

    # Cara superior:-------------------
    muro3 = sg.SceneGraphNode('muro_3')
    muro3.transform = tr.matmul([tr.translate(0, 0.2, -0.1),
                                 tr.rotationX(-np.pi / 2),
                                 tr.scale(1, escalaMuro, 1),
                                 tr.translate(0, 0.5 - escalaMuro - 0.037409, 0)])
    muro3.childs += [muro]

    # Lados:---------------------------
    ladoIzq = sg.SceneGraphNode('lado_izq')
    ladoIzq.transform = tr.matmul([tr.translate(0.5, 0, -0.15),
                                   tr.rotationY(-np.pi / 2),
                                   tr.uniformScale(escalaMuro),
                                   tr.translate(0, 0.5 - escalaMuro - 0.037409, 0)])
    ladoIzq.childs += [lado]

    ladoDer = sg.SceneGraphNode('lado_der')
    ladoDer.transform = np.array([
        # reflejamos en x (lo vimos en aux 4)
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    ladoDer.childs += [ladoIzq]

    # Unimos todas las partes:-------------
    nodoMuro = sg.SceneGraphNode('muro')
    nodoMuro.childs += [muro1,
                        muro2,
                        muro3,
                        ladoIzq,
                        ladoDer
                        ]

    return nodoMuro  # retornamos un muro completo de 5 lados (abierto por abajo)


# función createCarScene crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),
                                      tr.translate(0.056390, 0.037409, 0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),
                                      tr.translate(-0.060390, 0.037409, -0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),
                                      tr.translate(-0.056390, 0.037409, 0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),
                                      tr.translate(0.066090, 0.037409, -0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0),
                                tr.rotationY(np.pi)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    # se crea este nodo para almacenar solo la traslación en el movimiento
    carTranslation = sg.SceneGraphNode('car_translation')
    carTranslation.childs += [car1]

    scene = sg.SceneGraphNode('system')
    scene.childs += [carTranslation]

    return scene


# Función createStaticScene crea toda la escena estática y texturada de esta aplicación.
# Se utilizan las funciones createTexturedArc, createTiledFloor, createHouse y createWall
def createStaticScene(pipeline):
    # PRIMITIVAS:---------------------------------------
    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture

    # NODOS:---------------------------------------------
    # Piso:-----------------------
    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0, -0.1, 0.0)
    sandNode.childs += [sandBaseShape]

    # Camino:--------------------
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi / 2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    linearSector = sg.SceneGraphNode('linearSector')

    for i in range(10):
        node = sg.SceneGraphNode('road' + str(i) + '_ls')
        node.transform = tr.translate(0.0, 0.0, -1.0 * i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0, 0.0, -4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0, 0.0, 5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]

    # Casas:------------------------------------
    casa1 = sg.SceneGraphNode('casa_1')
    casa1.transform = tr.matmul([tr.translate(-3.5, 0, 0),
                                 tr.rotationY(np.pi / 2)])
    casa1.childs += [createHouse(pipeline, 'wall5.jpg', 'roof4.jpg', 'door4.jpg', 'window11.jpg')]  # creamos una casa

    # copiamos la casa anterior en otro lugar
    casa2 = sg.SceneGraphNode('casa_2')
    casa2.transform = tr.matmul([tr.translate(0, 0, 2),
                                 np.array([
                                     # reflejamos en x (lo vimos en aux 4)
                                     [-1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)])
    casa2.childs += [casa1]

    # copiamos la casa anterior en otro lugar
    casa3 = sg.SceneGraphNode('casa_3')
    casa3.transform = tr.matmul([tr.translate(0, 0, 0),
                                 np.array([
                                     # reflejamos en x (lo vimos en aux 4)
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)])
    casa3.childs += [casa2]

    casa4 = sg.SceneGraphNode('casa_4')
    casa4.transform = tr.matmul([tr.translate(3.5, 0, 0),
                                 tr.rotationY(-np.pi / 2)])
    casa4.childs += [createHouse(pipeline, 'wall4.jpg', 'roof3.jpg', 'door5.jpg', 'window12.jpg')]  # creamos otra casa

    # copiamos la casa anterior en otro lugar
    casa5 = sg.SceneGraphNode('casa_5')
    casa5.transform = tr.matmul([tr.translate(0, 0, 2),
                                 np.array([
                                     # reflejamos en x (lo vimos en aux 4)
                                     [-1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)])
    casa5.childs += [casa4]

    # copiamos la casa anterior en otro lugar
    casa6 = sg.SceneGraphNode('casa_6')
    casa6.transform = tr.matmul([tr.translate(0, 0, 0),
                                 np.array([
                                     # reflejamos en x (lo vimos en aux 4)
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32)])
    casa6.childs += [casa5]

    casa7 = sg.SceneGraphNode('casa_7')
    casa7.transform = tr.matmul([tr.translate(-3.5, 0, 4),
                                 np.array([
                                     # reflejamos en x y z (lo vimos en aux 4)
                                     [-1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]], dtype=np.float32),
                                 tr.rotationY(-np.pi / 2)])
    casa7.childs += [createHouse(pipeline, 'wall2.jpg', 'roof5.jpg', 'door5.jpg', 'window13.jpg')]  # creamos otra casa

    # copiamos la casa anterior en otro lugar
    casa8 = sg.SceneGraphNode('casa_8')
    casa8.transform = tr.matmul([np.array([
        # reflejamos en x (lo vimos en aux 4)
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)])
    casa8.childs += [casa7]

    # unimos todas las casas en un único nodo
    nodoCasas = sg.SceneGraphNode('casas')
    nodoCasas.childs += [casa1,
                         casa2,
                         casa3,
                         casa4,
                         casa5,
                         casa6,
                         casa7,
                         casa8
                         ]

    # Muros:------------------------------------
    # Barrera izquierda del mapa:
    muro0 = sg.SceneGraphNode('muro_0')
    muro0.transform = tr.rotationY(np.pi / 2)
    muro0.childs += [createWall(pipeline, 'wall3.jpg')]

    muro1 = sg.SceneGraphNode('muro_1')
    muro1.transform = tr.translate(-2.5, 0, 0)
    muro1.childs += [muro0]

    muro2 = sg.SceneGraphNode('muro_2')
    muro2.transform = tr.translate(0, 0, -1)
    muro2.childs += [muro1]

    muro3 = sg.SceneGraphNode('muro_3')
    muro3.transform = np.array([
        # reflejamos en z (lo vimos en aux 4)
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    muro3.childs += [muro2]

    muro4 = sg.SceneGraphNode('muro_4')
    muro4.transform = tr.matmul([tr.translate(0, 0, -1)])
    muro4.childs += [muro2]

    muro5 = sg.SceneGraphNode('muro_5')
    muro5.transform = np.array([
        # reflejamos en z (lo vimos en aux 4)
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    muro5.childs += [muro4]

    muro6 = sg.SceneGraphNode('muro_6')
    muro6.transform = tr.matmul([tr.translate(0, 0, -1)])
    muro6.childs += [muro4]

    muro7 = sg.SceneGraphNode('muro_7')
    muro7.transform = np.array([
        # reflejamos en z (lo vimos en aux 4)
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    muro7.childs += [muro6]

    muro8 = sg.SceneGraphNode('muro_8')
    muro8.transform = tr.matmul([tr.translate(0, 0, -1)])
    muro8.childs += [muro6]

    muro9 = sg.SceneGraphNode('muro_9')
    muro9.transform = np.array([
        # reflejamos en z (lo vimos en aux 4)
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    muro9.childs += [muro8]

    muro10 = sg.SceneGraphNode('muro_10')
    muro10.transform = tr.matmul([tr.translate(0, 0, 1)])
    muro10.childs += [muro9]

    # unimos los 10 muros izquierdos en un único nodo
    murosIzq = sg.SceneGraphNode('muros_izq')
    murosIzq.transform = tr.matmul([tr.translate(-1.7, 0, 0), tr.scale(1 / 3, 1, 1)])
    murosIzq.childs += [muro1,
                        muro2,
                        muro3,
                        muro4,
                        muro5,
                        muro6,
                        muro7,
                        muro8,
                        muro9,
                        muro10
                        ]

    # Barrera derecha del mapa: (reflejamos la barrera izquierda)
    murosDer = sg.SceneGraphNode('muros_der')
    murosDer.transform = np.array([
        # reflejamos en x (lo vimos en aux 4)
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    murosDer.childs += [murosIzq]

    # Unimos ambos muros en un único nodo:
    nodoMuros = sg.SceneGraphNode('muros')
    nodoMuros.childs += [murosDer, murosIzq]

    # Unimos toda la escena
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft,
                     linearSectorRight,
                     arcTop,
                     arcBottom,
                     sandNode,
                     nodoCasas,
                     nodoMuros
                     ]

    return scene


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 3"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()

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

    # Se crean los objetos de escena estática y el auto:
    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)

    setPlot(texPipeline, axisPipeline, lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    # BUCLE PRINCIPAL: -------------------------------------------

    # Parámetros iniciales:-------------------
    angle = 0.0
    aperturaRuedas = 0.07

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if controller.fillPolygon:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setView(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        # AGREGAMOS MOVIMIENTO AL AUTO:--------------------------

        # Parámetros de cambio:
        difAngle = 0.05
        distance = 0.05
        maximaAperturaRuedas = 0.4
        incrementoAperturaRuedas = 0.01

        # Buscamos los nodos a cambiar sus transformaciones:
        auto = sg.findNode(car, "car1")
        autoTranslate = sg.findNode(car, "car_translation")
        rueda = sg.findNode(car, 'rotatingWheel')

        # Definimos función que cambia la cámara:
        def changeCamera():
            # Cámara:
            posicionAuto = sg.findPosition(car, 'car1')                              # posición completa auto
            posicion = [posicionAuto[0, 0], posicionAuto[1, 0], posicionAuto[2, 0]]  # x, y y z de posición del auto
            cercania = 2                                                             # cercanía de la camara al auto
            altura = 1.1                                                             # altura de la cámara
            controller.viewPos = np.array([posicion[0]+cercania*np.sin(angle),       # nuevo eye
                                           posicion[1]+altura,
                                           posicion[2]+cercania*np.cos(angle)])
            controller.at = np.array([posicion[0], posicion[1], posicion[2]])        # nuevo at

        # Agregamos condiciones según las distintas keys:

        # Tecla A:
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            # Ruedas:
            if aperturaRuedas > maximaAperturaRuedas:
                aperturaRuedas -= incrementoAperturaRuedas
            rueda.transform = tr.rotationY(aperturaRuedas)
            if not(aperturaRuedas > maximaAperturaRuedas):
                aperturaRuedas += incrementoAperturaRuedas

            # Auto completo:
            angle += difAngle  # cambio ángulo de rotación debe ser positivo
            auto.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0),
                                        tr.rotationY(np.pi),
                                        tr.rotationY(angle)])
            # Cámara:
            changeCamera()

        # Tecla S:
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            #Ruedas:
            aperturaRuedas = 0.07 # reset apertura de ruedas
            rueda.transform = tr.rotationX(-angle * 10)
            #Auto completo:
            autoTranslate.transform = tr.matmul([tr.translate(distance * np.sin(angle), 0, distance * np.cos(angle)),
                                                 autoTranslate.transform])
            #Cámara:
            changeCamera()

        # Tecla D:
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            # Ruedas:
            if aperturaRuedas > maximaAperturaRuedas:
                aperturaRuedas -= incrementoAperturaRuedas
            rueda.transform = tr.rotationY(-aperturaRuedas)
            if not (aperturaRuedas > maximaAperturaRuedas):
                aperturaRuedas += incrementoAperturaRuedas

            # Auto completo:
            angle -= difAngle  # cambio ángulo de rotación debe ser negativo
            auto.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0),
                                        tr.rotationY(np.pi),
                                        tr.rotationY(angle)])

            # Cámara:
            changeCamera()

        # Tecla W:
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            # Ruedas:
            aperturaRuedas = 0.07  # reset apertura de ruedas
            rueda.transform = tr.rotationX(angle * 10)
            #Auto completo:
            distance = -distance  # distancia de mov negativa
            autoTranslate.transform = tr.matmul([tr.translate(distance * np.sin(angle), 0, distance * np.cos(angle)),
                                                 autoTranslate.transform])
            #Cámara:
            changeCamera()

        # DIBUJAMOS LA ESCENA:------------------------------------

        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")

        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    glfw.terminate()
