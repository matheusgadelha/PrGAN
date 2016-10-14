from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
#import matplotlib.image as mpimg
#import sys
#import argparse
#import ops


class Camera:
    def __init__(self, t=0, p=0.5, r=5.0):
        self.theta = t
        self.phi = p
        self.radius = r

    def place(self):
        px = self.radius * np.cos(self.theta) * np.cos(self.phi)
        py = self.radius * np.sin(self.phi)
        pz = self.radius * np.sin(self.theta) * np.cos(self.phi)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(px, py, pz, 0, 0, 0, 0, 1, 0)


def sphere_to_cartesian(array):
    px = array[2] * np.cos(array[0]) * np.cos(array[1])
    py = array[2] * np.sin(array[1])
    pz = array[2] * np.sin(array[0]) * np.cos(array[1])

    return np.array([px, py, pz])


def classic_sphere_to_cartesian(array):
    px = array[2] * np.cos(array[0]) * np.sin(array[1])
    py = array[2] * np.cos(array[1])
    pz = array[2] * np.sin(array[0]) * np.sin(array[1])

    return np.array([px, py, pz])


class RenderUtils:

    @staticmethod
    def vertex(v):
        glVertex3f(v[0], v[1], v[2])

    @staticmethod
    def normal(n):
        glNormal3f(n[0], n[1], n[2])

    @staticmethod
    def color(c):
        glColor3f(c[0], c[1], c[2])

    @staticmethod
    def color4(c):
        glColor4f(c[0], c[1], c[2], c[3])

    @staticmethod
    def draw_line(v1, v2):
        glBegin(GL_LINES)
        RenderUtils.vertex(v1)
        RenderUtils.vertex(v2)
        glEnd()

    @staticmethod
    def draw_points(s):
        glBegin(GL_POINTS)
        for i in s:
            RenderUtils.vertex(i)
        glEnd()


class Sphere(object):

    def __init__(self, radius=1.0, resolution=10):
        self.radius = radius
        self.resolution = resolution
        self.vertices = []
        self.colors = []

        self.grid = np.zeros((resolution, resolution, 3))

        self.vertices, self.colors = self.build_geometry()

    def build_geometry(self):

        theta_values = np.linspace(0, 2 * np.pi, self.resolution)
        phi_values = np.linspace(np.pi / 2., -np.pi / 2., self.resolution)

        vertices = []
        colors = []

        for p in range(self.resolution-1):
            for t in range(self.resolution):
                v0 = np.array([theta_values[t], phi_values[p], self.radius])
                v0 = sphere_to_cartesian(v0)
                vertices.append(v0)
                colors.append(np.random.rand(3))

                v1 = np.array([theta_values[t], phi_values[p+1], self.radius])
                v1 = sphere_to_cartesian(v1)
                vertices.append(v1)
                colors.append(np.random.rand(3))

                v2 = np.array([theta_values[(t+1) % self.resolution], phi_values[p], self.radius])
                v2 = sphere_to_cartesian(v2)
                vertices.append(v2)
                colors.append(np.random.rand(3))

                vertices.append(v1)
                colors.append(np.random.rand(3))

                v3 = np.array([theta_values[(t+1) % self.resolution], phi_values[p+1], self.radius])
                v3 = sphere_to_cartesian(v3)
                vertices.append(v3)
                colors.append(np.random.rand(3))

                vertices.append(v2)
                colors.append(np.random.rand(3))

        return vertices, colors

    def draw(self):
        RenderUtils.color([0, 0, 0])
        glBegin(GL_TRIANGLES)
        for i, v in enumerate(self.vertices):
            RenderUtils.color(self.colors[i])
            RenderUtils.vertex(v)
        glEnd()


class GLWindow(object):

    instance = None

    def __init__(self, window_name='OpenGL Window', window_size=(640, 640)):
        self.screen_size = window_size
        self.window_name = window_name
        GLWindow.instance = self

        glutInit()
        glutInitWindowSize(window_size[0], window_size[1])
        glutCreateWindow(self.window_name)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutDisplayFunc(GLWindow.displayWrapper)
        glutIdleFunc(GLWindow.displayWrapper)
        glutMouseFunc(GLWindow.mouseWrapper)
        glutMotionFunc(GLWindow.motionWrapper)

        self.initialize()

    def initialize(self):
        glClearDepth(1.0)
        glClearColor(1.0, 0.0, 0.0, 0.0)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def mouse(self, button, state, x, y):
        pass

    def motion(self, x, y):
        pass

    @staticmethod
    def get_instance(ss=(640, 640)):
        if GLWindow.instance is not None:
            return GLWindow.instance
        else:
            GLWindow.instance = GLWindow(ss)
            return GLWindow.instance

    @staticmethod
    def displayWrapper():
        GLWindow.get_instance().display()
        glutSwapBuffers()

    @staticmethod
    def mouseWrapper(button, state, x, y):
        GLWindow.get_instance().mouse(button, state, x, y)

    @staticmethod
    def motionWrapper(x, y):
        GLWindow.get_instance().motion(x, y)

    @staticmethod
    def initializeWrapper():
        GLWindow.get_instance().initialize()


def lerp(a, b, t):
    return a + (b-a)*t


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


if __name__ == '__main__':
    window = GLWindow()
