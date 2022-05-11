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

#TAREA4: Se crea un pool de texturas (diccionario). Las texturas se cargan una sola vez y se reutilizan cuando queramos.
texturesPool = dict()

#TAREA4: Esta función carga las texturas. Como el pool es un diccionario, cada textura se almacena con una clave única. 
# Por ejemplo: clave 'roof' para la textura del techo
def loadTextures():
    texturesPool['roof'] = es.textureSimpleSetup(getAssetPath("roof2.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR,GL_LINEAR)
    texturesPool['wallHouse'] = es.textureSimpleSetup(getAssetPath("wall5.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR,GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    texturesPool['wallWall'] = texturesPool['wallHouse']


def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

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

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

#TAREA4: Esta función se modificó para que cada vértice ahora contenga también las normales
def createTexturedArc(d):
    #TAREA4: En esta geometría, las normales siempre apuntan a Y+, por eso se agregó (0.0, 1.0, 0.0) como normales 
    # a todos los vértices generados en esta función
    vertices = [d, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4, 0.0, 1.0, 0.0]) #TAREA4: normal añadida
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4, 0.0, 1.0, 0.0]) #TAREA4: normal añadida
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4, 0.0, 1.0, 0.0]) #TAREA4: normal añadida
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4, 0.0, 1.0, 0.0]) #TAREA4: normal añadida

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

#TAREA4: Se modificó esta función para que los vértices contengan ahora sus normales
def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1, 0.0, 1.0, 0.0]) #TAREA4: normal añadida (siempre en Y+)
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1, 0.0, 1.0, 0.0]) #TAREA4: normal añadida (siempre en Y+)
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0, 0.0, 1.0, 0.0]) #TAREA4: normal añadida (siempre en Y+)
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0, 0.0, 1.0, 0.0]) #TAREA4: normal añadida (siempre en Y+)
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

# TAREA4: se modificó esta función para que las normales de cualquier plano siempre apunten hacia afuera de la geometría
# Recuerda que ahora tenemos iluminación y el sentido de las normales importa

def createHouse(pipeline):
    #Cuadrado centrado en el origen en el plano xy
    quadNode = sg.SceneGraphNode('system-house')
    quadNode.transform = tr.matmul([tr.translate(0.0,0.4,0.0)])

    #Agregamos la textura de la pared a un cuadrado
    quadWall = createGPUShape(pipeline, bs.createTextureQuadWithNormal(1,1))
    quadWall.texture = texturesPool['wallHouse'] #TAREA4: usamos la textura del pool
    
    #Las paredes formaran un cubo sin tapa inferior ni superior.
    #Pared basica 1
    Wall1 = sg.SceneGraphNode("Wall1")
    Wall1.transform = tr.matmul([tr.scale(1.0,0.8,1.0),tr.translate(0.0,0.0,0.5)])
    Wall1.childs += [quadWall]

    #Pared basica 2
    Wall2 = sg.SceneGraphNode("Wall2")
    Wall2.transform = tr.matmul([tr.scale(1.0,0.8,1.0),tr.translate(0.5,0.0,0.0),tr.rotationY(np.pi/2)])
    Wall2.childs += [quadWall]

    #Pared  basica 3
    Wall3 = sg.SceneGraphNode("Wall3")
    Wall3.transform = tr.matmul([tr.scale(1.0,0.8,1.0),tr.translate(0.0,0.0,-0.5),tr.rotationY(np.pi)])
    Wall3.childs += [quadWall]

    #Pared basica 4
    Wall4 = sg.SceneGraphNode("Wall4")
    Wall4.transform = tr.matmul([tr.scale(1.0,0.8,1.0),tr.translate(-0.5,0.0,0.0),tr.rotationY(-np.pi/2)])
    Wall4.childs += [quadWall]

    #Agregamos la textura del techo a un cuadrado
    quadRoof = createGPUShape(pipeline, bs.createTextureQuadWithNormal(1,1))
    quadRoof.texture = texturesPool['roof']
    
    #El techo estara conformado por dos rectangulos en diagonal con forma de V invertida.
    #techo 1
    Roof1 = sg.SceneGraphNode("Roof1")
    Roof1.transform = tr.matmul([tr.scale(1,0.4,1),tr.translate(0.0,1.2,-0.35),tr.rotationX(np.pi/4),tr.rotationY(np.pi)])
    Roof1.childs += [quadRoof]

    #techo 2
    Roof2 = sg.SceneGraphNode("Roof2")
    Roof2.transform = tr.matmul([tr.scale(1,0.4,1),tr.translate(0.0,1.2,0.35),tr.rotationX(-np.pi/4)])
    Roof2.childs += [quadRoof]

    #Grupo con todas las partes de la casa
    quadNode.childs += [Wall1]
    quadNode.childs += [Wall2]
    quadNode.childs += [Wall3]
    quadNode.childs += [Wall4]
    quadNode.childs += [Roof1]
    quadNode.childs += [Roof2]

    return quadNode #retorna un grafo de escena que tiene la geometria de una casa centrada en el origen.

#TAREA4: Se modificó esta función para que el muro de contención tenga caras que siempre apuntan hacia afuera
def createWall(pipeline):
    #rectangulo centrado en el origen en el plano xy
    quadNode = sg.SceneGraphNode('system-wall')
    quadNode.transform = tr.matmul([tr.scale(5,0.4,0.2), tr.translate(0.0,0.5,0.0)])
    
    #Agregamos la textura de la pared a un cuadrado
    quadWall = createGPUShape(pipeline, bs.createTextureQuadWithNormal(1,1))
    quadWall.texture = texturesPool['wallWall'] #TAREA4: Usamos la textura del pool
    
    wall1 = sg.SceneGraphNode("Muralla_z+")
    wall1.transform = tr.translate(0.0, 0.0, 0.5)
    wall1.childs += [quadWall]

    wall2 = sg.SceneGraphNode("Muralla_z-")
    wall2.transform = tr.matmul([tr.translate(0.0, 0.0, -0.5), tr.rotationY(np.pi)])
    wall2.childs += [quadWall]

    wall3 = sg.SceneGraphNode("Muralla_x+")
    wall3.transform = tr.matmul([tr.translate(0.5, 0.0, 0.0), tr.rotationY(np.pi/2)])
    wall3.childs += [quadWall]

    wall4 = sg.SceneGraphNode("Muralla_x-")
    wall4.transform = tr.matmul([tr.translate(-0.5, 0.0, 0.0), tr.rotationY(-np.pi/2)])
    wall4.childs += [quadWall]

    wall5 = sg.SceneGraphNode("Muralla_y+")
    wall5.transform = tr.matmul([tr.translate(0.0, 0.5, 0.0), tr.rotationX(-np.pi/2)])
    wall5.childs += [quadWall]

    wall6 = sg.SceneGraphNode("Muralla_y-")
    wall6.transform = tr.matmul([tr.translate(0.0, -0.5, 0.0), tr.rotationX(np.pi/2)])
    wall6.childs += [quadWall]
    
    #Grupo con todas las murallas
    quadNode.childs += [wall1]
    quadNode.childs += [wall2]
    quadNode.childs += [wall3]
    quadNode.childs += [wall4]
    quadNode.childs += [wall5]
    quadNode.childs += [wall6]


    return quadNode #retorna un grafo de escena que tiene la geometria de un par de murallas (una a cada lado de la pista) (no centradas en el origen)

# TAREA3: Esta función crea un grafo de escena especial para el auto.
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
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0), tr.rotationY(np.pi)]) 
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system-car')
    scene.childs += [car1]

    return scene

# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista

#Tarea4: arreglar para funcionar con texturas y normales
def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuadWithNormal(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture
    
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]
    
                        ############################## CASAS ##############################
    #creamos una casa centrada en el origen
    house = createHouse(pipeline)

    #Creamos 10 casas a lo largo de la pista (5 a cada lado)
    house1 = sg.SceneGraphNode('house1')
    house1.transform = tr.translate(4.0,0,2.0)
    house1.childs += [house]

    house2 = sg.SceneGraphNode('house2')
    house2.transform = tr.translate(-4.0,0,2.0)
    house2.childs += [house]

    house3 = sg.SceneGraphNode('house3')
    house3.transform = tr.translate(4.0,0,4.0)
    house3.childs += [house]

    house4 = sg.SceneGraphNode('house4')
    house4.transform = tr.translate(-4.0,0,4.0)
    house4.childs += [house]

    house5 = sg.SceneGraphNode('house5')
    house5.transform = tr.translate(4.0,0,-2.0)
    house5.childs += [house]

    house6 = sg.SceneGraphNode('house6')
    house6.transform = tr.translate(-4.0,0,-2.0)
    house6.childs += [house]

    house7 = sg.SceneGraphNode('house7')
    house7.transform = tr.translate(4.0,0,-4.0)
    house7.childs += [house]

    house8 = sg.SceneGraphNode('house8')
    house8.transform = tr.translate(-4.0,0,-4.0)
    house8.childs += [house]

    house9 = sg.SceneGraphNode('house9')
    house9.transform = tr.translate(4.0,0,0)
    house9.childs += [house]

    house10 = sg.SceneGraphNode('house10')
    house10.transform = tr.translate(-4.0,0,0)
    house10.childs += [house]

    #grupo con todas las casas
    houses = sg.SceneGraphNode('houses')
    houses.childs += [house1] 
    houses.childs += [house2] 
    houses.childs += [house3] 
    houses.childs += [house4]
    houses.childs += [house5] 
    houses.childs += [house6]  
    houses.childs += [house7] 
    houses.childs += [house8]
    houses.childs += [house9] 
    houses.childs += [house10]
    
                ############################## MURALLAS DE CONTENCIÓN ##############################

    wall = createWall(pipeline)
    #creamos 4 pares de murallas (2 a cada lado)
    wall1 = sg.SceneGraphNode('wall1')
    wall1.transform = tr.matmul([tr.translate(2.6,0,0), tr.rotationY(np.pi/2)])
    wall1.childs += [wall]

    wall2 = sg.SceneGraphNode('wall2')
    wall2.transform = tr.matmul([tr.translate(-2.6,0,0), tr.rotationY(np.pi/2)])
    wall2.childs += [wall]

    wall3 = sg.SceneGraphNode('wall3')
    wall3.transform = tr.matmul([tr.translate(1.4,0,0), tr.rotationY(np.pi/2)])
    wall3.childs += [wall]

    wall4 = sg.SceneGraphNode('wall4')
    wall4.transform = tr.matmul([tr.translate(-1.4,0,0), tr.rotationY(np.pi/2)])
    wall4.childs += [wall]
    
    #grupo con todas las murallas
    walls = sg.SceneGraphNode('walls')
    walls.childs += [wall1]
    walls.childs += [wall2]
    walls.childs += [wall3]
    walls.childs += [wall4]

    scene = sg.SceneGraphNode('system-static')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    scene.childs += [houses] #se agregan las casas a la escena
    scene.childs += [walls] #se agregan las murallas a la escena
    
    return scene