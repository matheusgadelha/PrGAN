from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import Sphere
from renderutils import RenderUtils
from scipy.special import sph_harm
from mesh import Mesh

import numpy as np
import renderutils
#import matplotlib.image as mpimg
#import sys
#import argparse
#import ops


class SphericalHarmonicsMesh(Sphere):
    def __init__(self, radius=2.0, resolution=50, l=2, m=1):
        self.l = l
        self.m = m
        super(SphericalHarmonicsMesh, self).__init__(radius, resolution)

        self.vertices, self.colors = self.build_geometry()

    def build_geometry(self):
        theta_values = np.linspace(0, 2 * np.pi, self.resolution)
        phi_values = np.linspace(-np.pi/2., np.pi/2., self.resolution)

        vertices = []
        colors = []

        for p in range(self.resolution - 1):
            for t in range(self.resolution):
                v0, c0 = self.create_sh_vertex(theta_values[t], phi_values[p])
                vertices.append(v0)
                colors.append(c0)

                v1, c1 = self.create_sh_vertex(theta_values[t], phi_values[p + 1])
                vertices.append(v1)
                colors.append(c1)

                v2, c2 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p])
                vertices.append(v2)
                colors.append(c2)

                vertices.append(v1)
                colors.append(c1)

                v3, c3 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p+1])
                vertices.append(v3)
                colors.append(c3)

                vertices.append(v2)
                colors.append(c2)

        return vertices, colors

    def create_sh_vertex(self, theta, phi):
        r = sph_harm(self.m, self.l, theta, phi+np.pi/2.).real * self.radius
        vertex = np.array([theta, phi, np.abs(r)])

        return renderutils.sphere_to_cartesian(vertex),\
               renderutils.lerp(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (r/self.radius + 1)/2.)


class SphericalHarmonicsViewer(GLWindow):

    def __init__(self, window_name='Spherical Harmonics Viewer', window_size=(640, 640)):
        super(SphericalHarmonicsViewer, self).__init__(window_name, window_size)
        self.sh = SphericalHarmonicsMesh()
        self.camera = Camera()
        self.mesh = Mesh(os.path.join("..", "models", "cube.obj"))
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
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.sh.draw()
        # RenderUtils.color([0, 0, 0])
        # self.mesh.draw()

if __name__ == '__main__':
    window = SphericalHarmonicsViewer()