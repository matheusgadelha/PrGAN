from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import Sphere
from renderutils import sphere_to_cartesian

from scipy.special import sph_harm

import numpy as np
#import matplotlib.image as mpimg
#import sys
#import argparse
#import ops

class SphericalHarmonicsMesh(Sphere):
    def __init__(self, radius=2.0, resolution=50, l=7, m=5):
        self.l = l
        self.m = m
        super(SphericalHarmonicsMesh, self).__init__(radius, resolution)

        self.vertices, self.colors = self.build_geometry()

    def build_geometry(self):
        theta_values = np.linspace(0, 2 * np.pi, self.resolution)
        phi_values = np.linspace(np.pi / 2., -np.pi / 2., self.resolution)

        vertices = []
        colors = []

        for p in range(self.resolution - 1):
            for t in range(self.resolution):
                v0 = self.create_sh_vertex(theta_values[t], phi_values[p])
                vertices.append(v0)
                colors.append(np.random.rand(3))

                v1 = self.create_sh_vertex(theta_values[t], phi_values[p + 1])
                vertices.append(v1)
                colors.append(np.random.rand(3))

                v2 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p])
                vertices.append(v2)
                colors.append(np.random.rand(3))

                vertices.append(v1)
                colors.append(np.random.rand(3))

                v3 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p+1])
                vertices.append(v3)
                colors.append(np.random.rand(3))

                vertices.append(v2)
                colors.append(np.random.rand(3))

        return vertices, colors

    def create_sh_vertex(self, theta, phi):
        r = sph_harm(self.m, self.l, theta, phi).real * self.radius
        vertex = np.array([theta, phi, r])
        return sphere_to_cartesian(vertex)


class SphericalHarmonicsViewer(GLWindow):

    def __init__(self, window_name='Spherical Harmonics Viewer', window_size=(640, 640)):
        super(SphericalHarmonicsViewer, self).__init__(window_name, window_size)
        self.sh = SphericalHarmonicsMesh()
        self.camera = Camera()
        self.camera_speed = 1/100.
        self.initialize()
        self.action = ""

        self.prev_x = 0
        self.prev_y = 0

        glutMainLoop()

    def mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            self.action = "MOVE_CAMERA"

    def motion(self, x, y):
        if self.action == "MOVE_CAMERA":
            dx = self.prev_x - x
            dy = self.prev_y - y
            self.camera.theta -= dx * self.camera_speed
            self.camera.phi += dy * self.camera_speed
            self.prev_x = x
            self.prev_y = y

    def initialize(self):
        MAX_COORD = 2.

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClearDepth(1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-MAX_COORD, MAX_COORD, -MAX_COORD, MAX_COORD,
                0.1, 1000.0)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.place()
        self.sh.draw()

if __name__ == '__main__':
    window = SphericalHarmonicsViewer()