from OpenGL.GL import *
from OpenGL.GLUT import *
from renderutils import RenderUtils

import numpy as np


class Mesh(object):

    def __init__(self, path):
        self.vertices = []
        self.normals = []
        self.indices = []

        for line in open(path, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = np.array([float(v) for v in values[1:4]])
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = np.array([float(v) for v in values[1:4]])
                self.normals.append(v)
            elif values[0] == 'f':
                self.indices.append(int(values[1].split("/")[0]))
                self.indices.append(int(values[2].split("/")[0]))
                self.indices.append(int(values[3].split("/")[0]))

    def draw(self):
        glBegin(GL_TRIANGLES)
        for i in self.indices:
            RenderUtils.vertex(self.vertices[i-1])
        glEnd()
