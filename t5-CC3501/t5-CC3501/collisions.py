# coding=utf-8
"""Funciones y clases para implementar colisiones AABB"""

from typing import List
import numpy as np
from OpenGL.GL import *
import math
import my_scenes as msc
import grafica.transformations as tr
import grafica.basic_shapes as bs

__author__ = "Sebastian Olmos"
__license__ = "MIT"


# Funcion para crear la GPUShape de lineas de un cubo
def createBoundingBox(color):
    # Defining the location and colors of each vertex  of the shape
    vertices = [
        #    positions         colors
        -0.5, -0.5, 0.5, color[0], color[1], color[2],
        0.5, -0.5, 0.5, color[0], color[1], color[2],
        0.5, 0.5, 0.5, color[0], color[1], color[2],
        -0.5, 0.5, 0.5, color[0], color[1], color[2],

        -0.5, -0.5, -0.5, color[0], color[1], color[2],
        0.5, -0.5, -0.5, color[0], color[1], color[2],
        0.5, 0.5, -0.5, color[0], color[1], color[2],
        -0.5, 0.5, -0.5, color[0], color[1], color[2]]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 1, 2, 2, 3, 3, 0,
        0, 4, 1, 5, 2, 6, 3, 7,
        4, 5, 5, 6, 6, 7, 7, 4]

    return bs.Shape(vertices, indices)


# Clase para la repreesentacion aabb de un objeto
class AABB:
    # Constructor Que recibe como parametros:
    #    center -> lista de tres valores que representan las coordenadas de mundo del centro del objeto
    #    distX  -> distancia desde el centro a la bounding box en el eje x
    #    distY  -> distancia desde el centro a la bounding box en el eje y
    #    distZ  -> distancia desde el centro a la bounding box en el eje z
    def __init__(self, center: List[float], distX: float, distY: float, distZ: float) -> None:
        self.distX = distX
        self.distY = distY
        self.distZ = distZ
        self.scale = tr.scale(distX * 2, distY * 2,
                              distZ * 2)  # transformacion de scala para la gpuShape del Bounding Box
        self.set_pos(center)

    def set_pos(self, center: List[float]):
        # Valores que definen la bounding box
        self.minX = center[0] - self.distX
        self.maxX = center[0] + self.distX
        self.minY = center[1] - self.distY
        self.maxY = center[1] + self.distY
        self.minZ = center[2] - self.distZ
        self.maxZ = center[2] + self.distZ
        self.pos = tr.translate(center[0], center[1],
                                center[2])  # transformacion de traslacion para la gpuShape del Bounding Box
        self.transform = tr.matmul([self.pos, self.scale])  # Transformacion del bounding box

    # Funcion que dado otro AABB verifica si estan colisionando
    def overlaps(self, other: "AABB") -> bool:
        return (self.maxX > other.minX and self.minX < other.maxX and
                self.maxY > other.minY and self.minY < other.maxY and
                self.maxZ > other.minZ and self.minZ < other.maxZ)

    # Funcion que dado otro AABB verifica si estan colisionando y retorna un vector indicando en que eje
    # Se debe anular un movimiento
    #    -> [1, 1, 1] : No hay colision, el objeto other puede moverse libremente en las tres coordenadas
    #    -> [1, 0, 1] : Hay colision, el objeto other puede moverse libremente en las coordenadas x, z
    def collide_and_slide(self, other: "AABB"):
        if self.overlaps(other):
            # si hay colision, se calcula las dimensiones de la caja que resulta al intersectar ambas aabb
            dx = min(self.maxX, other.maxX) - max(self.minX, other.minX)
            dy = min(self.maxY, other.maxY) - max(self.minY, other.minY)
            dz = min(self.maxZ, other.maxZ) - max(self.minZ, other.minZ)

            # Se ubica un 0 en el eje que presenta la dimension más pequeña

            # Caso en que la colision no permita moverse en el eje X
            if (math.fabs(dx) <= math.fabs(dy) and math.fabs(dx) <= math.fabs(dz)):
                return np.array([0, 1, 1])

            elif (math.fabs(dy) <= math.fabs(dx) and math.fabs(dy) <= math.fabs(dz)):
                # Caso en que la colision no permita moverse en el eje Y
                return np.array([1, 0, 1])

            elif (math.fabs(dz) <= math.fabs(dy) and math.fabs(dz) <= math.fabs(dx)):
                # Caso en que la colision no permita moverse en el eje Z
                return np.array([1, 1, 0])
        else:
            # Caso que no se detecta colision
            return np.array([1, 1, 1])

    # Funcion que dado otro AABB verifica si la prpoia aabb de este objeto se encuentra debajo del aabb que realiza la consulta
    def down_raycast(self, other: "AABB") -> bool:
        center_x = (other.minX + other.maxX) / 2
        center_z = (other.minZ + other.maxZ) / 2

        return (center_x > self.minX and center_x < self.maxX and center_z > self.minZ and center_z < self.maxZ and
                ((other.minY < self.maxY and other.minY > self.minY) or math.fabs(other.minY - self.minY) < 0.01))


# Clase para la estructura que almacena los objetos representados como aabb
class AABBList:
    def __init__(self, pipeline, color) -> None:
        self.objects: List[AABB] = []  # Lista de aabbs
        self.gpuBox = msc.createGPUShape(pipeline, createBoundingBox(color))  # GPUShape del bounding box

    # Funcion que dibuja la visualizacion de los bounding boces
    def drawBoundingBoxes(self, pipeline):
        glLineWidth(3)  # Se aumenta el tamaño de las lineas dibujadas
        for aabb in self.objects:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, aabb.transform)
            pipeline.drawCall(self.gpuBox, GL_LINES)
        glLineWidth(1)  # Se vuelve al tamaño normal de lineas

    # Consulta en todas las AABBs contenidas si existe colision con los AABB de other_list
    def check_overlap(self, other_list: "AABBList") -> bool:
        for box1 in self.objects:
            for box2 in other_list.objects:
                if box2.overlaps(box1):
                    return True

    # Consulta en todas las AABBs contenidas si existe colision con los AABB de other_list
    # Entregando un vector indicando en que eje se debe anular el movimiento
    #    -> [1, 1, 1] : No hay colision, el objeto other puede moverse libremente en las tres coordenadas
    #    -> [1, 0, 1] : Hay colision, el objeto other puede moverse libremente en las coordenadas x, z
    #    -> [0, 0, 0] : Hay colision, el objeto no puede moverse en ninguna coordenada
    def collide_and_slide(self, other_list: "AABBList") -> bool:
        direction = np.array([1, 1, 1])
        for box1 in self.objects:
            for box2 in other_list.objects:
                direction = direction * box1.collide_and_slide(box2)
        return direction

    # Consulta en todas las AABBs contenidas colisiones por debajo con el AABB de other
    def down_raycast(self, other: "AABB"):
        for box1 in self.objects:
            if box1.down_raycast(other):
                return True