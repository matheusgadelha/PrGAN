from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import Sphere

from scipy.special import sph_harm

import numpy as np
#import matplotlib.image as mpimg
#import sys
#import argparse
#import ops

#class SphericalHarmonicsMesh(Sphere):
#    def __init__(self, radius=1.0, resolution=50):
#        def

class SphericalHarmonicsViewer(GLWindow):

    def __init__(self, window_name='Spherical Harmonics Viewer', window_size=(640, 640)):
        super(SphericalHarmonicsViewer, self).__init__(window_name, window_size)
        self.sphere = Sphere(resolution=30)
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
        self.sphere.draw()

if __name__ == '__main__':
    window = SphericalHarmonicsViewer()