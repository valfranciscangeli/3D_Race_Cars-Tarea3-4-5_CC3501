# coding=utf-8
"""Funciones para crear las escenas del aux 8 + creacion de skybox"""

from os import pipe
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.scene_graph as sg
from grafica.assets_path import getAssetPath

__author__ = "Sebastian Olmos"
__license__ = "MIT"

# Convenience function to ease initialization
def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)
    return gpuShape

# Función para crear un árbol
def createTree(pipeline):
    # Creacion de primitivas en GPU
    brownCylinder = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.596, 0.447, 0.227))
    greenCone = createGPUShape(pipeline, bs.createColorConeTarea2(0.113, 0.713, 0.047))

    # Tronco del nodo
    trunkNode = sg.SceneGraphNode('trunk')
    trunkNode.transform = tr.matmul([tr.translate(0.0, 0.0, 0.0) , tr.scale(0.2, 0.4, 0.2)])
    trunkNode.childs += [brownCylinder]

    # Tronco de las hojas
    leafNode = sg.SceneGraphNode('leaf')
    leafNode.transform = tr.matmul([tr.translate(0.0, 0.8, 0.0) , tr.scale(0.5, 0.5, 0.5)])
    leafNode.childs += [greenCone]

    # Tronco del arbol a retornar
    treeNode = sg.SceneGraphNode('tree')
    treeNode.transform = tr.matmul([tr.translate(0.0, 0.5, 0.0) , tr.scale(1.0, 1.0, 1.0)])
    treeNode.childs += [trunkNode, leafNode]

    return treeNode

# Función para crear un templo
def createTemple(pipeline):
    # Creacion de primitivas
    cube1 = createGPUShape(pipeline, bs.createColorCubeTarea2(0.768, 0.690, 0.552))
    cube2 = createGPUShape(pipeline, bs.createColorCubeTarea2(0.890, 0.823, 0.709))
    cube3 = createGPUShape(pipeline, bs.createColorCubeTarea2(0.839, 0.803, 0.741))
    cylinder = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.847, 0.768, 0.674))

    # Nodo del primer bloque
    slab1Node = sg.SceneGraphNode('slab1')
    slab1Node.transform = tr.scale(1.0, 0.1, 0.7)
    slab1Node.childs += [cube1]

    # Nodo del segundo bloque
    slab2Node = sg.SceneGraphNode('slab2')
    slab2Node.transform = tr.scale(0.9, 0.1, 0.6)
    slab2Node.childs += [cube2]

    # Nodo del tercer bloque
    slab3Node = sg.SceneGraphNode('slab3')
    slab3Node.transform = tr.scale(0.8, 0.1, 0.5)
    slab3Node.childs += [cube3]

    # Nodo del grupo 3 que contiene al bloque 3 y el techo
    group3Node= sg.SceneGraphNode('group3') 
    group3Node.transform = tr.translate(0.0, 0.1, 0.0)
    group3Node.childs += [slab3Node]

    # Nodo del grupo 2 que contiene al bloque 2 y al grupo 3
    group2Node= sg.SceneGraphNode('group2')
    group2Node.transform = tr.translate(0.0, 0.1, 0.0)
    group2Node.childs += [slab2Node]
    group2Node.childs += [group3Node]
    
    # Nodo del grupo 1 que contiene al bloque 1 y al grupo 2
    group1Node= sg.SceneGraphNode('group1')
    group1Node.childs += [slab1Node]
    group1Node.childs += [group2Node]

    # Nodo de la primera columna
    column1Node = sg.SceneGraphNode('Column1')
    column1Node.transform = tr.matmul([tr.translate(0.6, 0.5, 0.0) , tr.scale(0.07, 0.4, 0.07)])
    column1Node.childs += [cylinder]

    # Nodo de la segunda columna
    column2Node = sg.SceneGraphNode('Column2')
    column2Node.transform = tr.matmul([tr.translate(0.2, 0.5, 0.0) , tr.scale(0.07, 0.4, 0.07)])
    column2Node.childs += [cylinder]

    # Nodo de la tercera columna
    column3Node = sg.SceneGraphNode('Column3')
    column3Node.transform = tr.matmul([tr.translate(-0.2, 0.5, 0.0) , tr.scale(0.07, 0.4, 0.07)])
    column3Node.childs += [cylinder]

    # Nodo de la cuarta columna
    column4Node = sg.SceneGraphNode('Column4')
    column4Node.transform = tr.matmul([tr.translate(-0.6, 0.5, 0.0) , tr.scale(0.07, 0.4, 0.07)])
    column4Node.childs += [cylinder]

    # Nodo que agrupa las columnas
    columnsNode = sg.SceneGraphNode('Columns')
    columnsNode.childs += [column1Node, column2Node, column3Node, column4Node]

    # Nodo de las columnas del lado izquierdo
    leftColumnsNode= sg.SceneGraphNode('leftColumns')
    leftColumnsNode.transform = tr.matmul([tr.translate(0.0, 0.0, 0.35) , tr.scale(1.0, 1.0, 1.0)])
    leftColumnsNode.childs += [columnsNode]

    # Nodo de las columnas del lado derecho
    rightColumnsNode= sg.SceneGraphNode('rightColumns')
    rightColumnsNode.transform = tr.matmul([tr.translate(0.0, 0.0, -0.35) , tr.scale(1.0, 1.0, 1.0)])
    rightColumnsNode.childs += [columnsNode]

    # Nodo que agrupa las columnas
    columnGroupNode= sg.SceneGraphNode('columnGroup')
    columnGroupNode.childs += [leftColumnsNode, rightColumnsNode]

    # Agregamos las columnas al grupo 3
    group3Node.childs += [columnGroupNode]

    # Nodo de la base del techo, encima de las columnas
    baseNode = sg.SceneGraphNode('base')
    baseNode.transform =  tr.scale(0.75, 0.05, 0.45)
    baseNode.childs += [cube1]

    # Nodo de la pendiente superior 2 del techo
    roof1Node = sg.SceneGraphNode('roof2')
    roof1Node.transform = tr.matmul([tr.translate(0.0, 0.2, 0.25) , tr.rotationX(np.pi/6), tr.scale(0.75, 0.05, 0.31)])
    roof1Node.childs += [cube2]

    # Nodo de la pendiente superior 1 del techo
    roof2Node = sg.SceneGraphNode('roof2')
    roof2Node.transform = tr.matmul([tr.translate(0.01, 0.2, -0.25) , tr.rotationX(-np.pi/6), tr.scale(0.75, 0.05, 0.31)])
    roof2Node.childs += [cube2]

    # Nodo del techo
    ceilNode = sg.SceneGraphNode('ceil')
    ceilNode.transform = tr.translate(0.0, 0.9, 0.0) , tr.scale(1.0, 1.0, 1.0)
    ceilNode.childs += [baseNode, roof1Node, roof2Node]

    # Agregamos el techo al nodo 3
    group3Node.childs += [ceilNode]
    
    # Creamos el nodo final 
    structNode = sg.SceneGraphNode('temple')
    structNode.transform = tr.uniformScale(2.0)
    structNode.childs += [group1Node]

    return structNode

# Funcion para crear el grafo de la escena
def createScene(pipeline):
    
    # Se crean las primitivas
    greenCube = createGPUShape(pipeline, bs.createColorCubeTarea2(0.572, 0.937, 0.121))
    yellowSphere = createGPUShape(pipeline, bs.createColorSphereTarea2(1.0, 1.0, 0.0))
    orangePlane = createGPUShape(pipeline, readOFF(getAssetPath('avion.off'), (1.0, 0.4, 0.0)))

    # Creamos el grafo de un árbol
    treeNode = createTree(pipeline)

    # Nodo del primer arbol
    tree1Node = sg.SceneGraphNode('tree1')
    tree1Node.transform = tr.matmul([tr.translate(5.0, 0.0, -4.0) , tr.scale(0.7, 0.7, 0.7)])
    tree1Node.childs += [treeNode]

    # Nodo del segundo arbol
    tree2Node = sg.SceneGraphNode('tree2')
    tree2Node.transform = tr.matmul([tr.translate(3.0, 0.0, 2.0) , tr.scale(1.2, 1.7, 1.2)])
    tree2Node.childs += [treeNode]

    # Nodo del tercer arbol
    tree3Node = sg.SceneGraphNode('tree3')
    tree3Node.transform = tr.matmul([tr.translate(-4.0, 0.0, -3.0) , tr.scale(1.3, 1.0, 1.3)])
    tree3Node.childs += [treeNode]

    # Nodo del cuarto arbol
    tree4Node = sg.SceneGraphNode('tree4')
    tree4Node.transform = tr.matmul([tr.translate(-6.0, 0.0, 4.0) , tr.scale(1.4, 1.2, 1.4)])
    tree4Node.childs += [treeNode]

    # Nodo del quinto arbol
    tree5Node = sg.SceneGraphNode('tree5')
    tree5Node.transform = tr.matmul([tr.translate(-5.0, 0.0, 7.0) , tr.scale(1.0, 1.0, 1.0)])
    tree5Node.childs += [treeNode]

    # Nodo del sexto arbol
    tree6Node = sg.SceneGraphNode('tree6')
    tree6Node.transform = tr.matmul([tr.translate(0.0, -3.0, -5.0) , tr.scale(3.0, 5.0, 3.0)])
    tree6Node.childs += [treeNode]

    # Nodo que contiene los arboles
    forestNode = sg.SceneGraphNode('forest')
    forestNode.childs += [tree1Node, tree2Node, tree3Node, tree4Node, tree5Node, tree6Node]

    # Nodo del suelo
    grassNode = sg.SceneGraphNode('grass')
    grassNode.transform = tr.matmul([tr.translate(0.0, -0.1, 0.0) , tr.scale(10.0, 0.1, 10.0)])
    grassNode.childs += [greenCube]

    # Nodo del avion
    planeNode = sg.SceneGraphNode('plane')
    planeNode.transform = tr.matmul([tr.translate(0.0, 0.7, 6.0) , tr.scale(0.3, 0.3, 0.3), tr.rotationY(np.pi/4), tr.rotationX(np.pi/12)])
    planeNode.childs += [orangePlane]

    # Nodo padre de la escena
    scene = sg.SceneGraphNode('scene')
    scene.childs += [grassNode]
    scene.childs += [createTemple(pipeline)]
    scene.childs += [forestNode]
    scene.childs += [planeNode]

    # Nodo de esfera
    sphere1Node = sg.SceneGraphNode('sphere1')
    sphere1Node.transform = tr.uniformScale(0.2)
    sphere1Node.childs += [yellowSphere]

    # Nodo para mover la esfera
    spherePosNode = sg.SceneGraphNode('spherePos')
    spherePosNode.childs += [sphere1Node]

    # Nodo para mover la esfera
    spherePos2Node = sg.SceneGraphNode('spherePos2')
    spherePos2Node.transform = tr.translate(0,-3, 0)
    spherePos2Node.childs += [sphere1Node]

    # Nodo para mover la esfera
    spherePos3Node = sg.SceneGraphNode('spherePos3')
    spherePos3Node.transform = tr.translate(0,-3, 0)
    spherePos3Node.childs += [sphere1Node]

    # Nodo para mover la esfera
    spherePos4Node = sg.SceneGraphNode('spherePos4')
    spherePos4Node.transform = tr.translate(0,-3, 0)
    spherePos4Node.childs += [sphere1Node]

    scene.childs += [spherePosNode]
    scene.childs += [spherePos2Node]
    scene.childs += [spherePos3Node]
    scene.childs += [spherePos4Node]
    
    return scene

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

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

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

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
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

#funcion para crear un cubo/ skybox con texturas
def createSkyBox(distance):
    dx = 1.0 / 4.0
    dy = 1.0 / 3.0
    r = distance
    vertices = []
    indices = []

    vertices += [
        -r, r, -r, 0 * dx, 2 * dy,
        r, r, -r, 1 * dx, 2 * dy,
        r, r, r, 1 * dx, 1 * dy,
        -r, r, r, 0 * dx, 1 * dy]
    indices += [0, 1, 2, 2, 3, 0]

    vertices += [
        r, r, -r, 1 * dx, 2 * dy,
        r, -r, -r, 2 * dx, 2 * dy,
        r, -r, r, 2 * dx, 1 * dy,
        r, r, r, 1 * dx, 1 * dy]
    indices += [4, 5, 6, 6, 7, 4]

    vertices += [
        r, -r, -r, 2 * dx, 2 * dy,
        -r, -r, -r, 3 * dx, 2 * dy,
        -r, -r, r, 3 * dx, 1 * dy,
        r, -r, r, 2 * dx, 1 * dy]
    indices += [8, 9, 10, 10, 11, 8]

    vertices += [
        -r, -r, -r, 3 * dx, 2 * dy,
        -r, r, -r, 4 * dx, 2 * dy,
        -r, r, r, 4 * dx, 1 * dy,
        -r, -r, r, 3 * dx, 1 * dy]
    indices += [12, 13, 14, 14, 15, 12]

    vertices += [
        -r, r, -r, 1 * dx, 3 * dy,
        -r, -r, -r, 2 * dx, 3 * dy,
        r, -r, -r, 2 * dx, 2 * dy,
        r, r, -r, 1 * dx, 2 * dy]
    indices += [16, 17, 18, 18, 19, 16]

    vertices += [
        r, r, r, 1 * dx, 1 * dy,
        r, -r, r, 2 * dx, 1 * dy,
        -r, -r, r, 2 * dx, 0 * dy,
        -r, r, r, 1 * dx, 0 * dy]
    indices += [20, 21, 22, 22, 23, 20]

    return bs.Shape(vertices, indices) 