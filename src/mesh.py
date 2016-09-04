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
                self.indices.append(int(values[1].split("/")[0])-1)
                self.indices.append(int(values[2].split("/")[0])-1)
                self.indices.append(int(values[3].split("/")[0])-1)

    def draw(self):
        glBegin(GL_TRIANGLES)
        for i in self.indices:
            RenderUtils.vertex(self.vertices[i])
        glEnd()

    def get_triangle(self, i):
        vertices = [self.vertices[self.indices[3 * i]],
                    self.vertices[self.indices[3 * i + 1]],
                    self.vertices[self.indices[3 * i + 2]]]
        return vertices


class Ray(object):

    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction

    def intersect_triangle(self, triangle):
        p0 = triangle[0]
        p1 = triangle[1]
        p2 = triangle[2]

        e1 = p1 - p0
        e2 = p2 - p0

        h = np.cross(self.d, e2)
        a = np.dot(e1, h)

        if -1e-6 < a < 1e-6:
            return None, None

        f = 1./a
        s = self.o - p0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None, None

        q = np.cross(s, e1)
        v = f * np.dot(self.d, q)
        if v < 0.0 or u + v > 1.0:
            return None, None

        t = f * np.dot(e2, q)

        if t < 1e-6:
            return None, t

        return self.o + t*self.d, t

    def intersect_mesh(self, mesh):
        n_triangles = len(mesh.indices)/3
        t_min = float('inf')
        intersection = None
        for i in range(n_triangles):
            point, t = self.intersect_triangle(mesh.get_triangle(i))
            if point is not None and t < t_min:
                t_min = t
                intersection = point

        return intersection



