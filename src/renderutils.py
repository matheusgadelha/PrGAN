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

        GLWindow.initializeWrapper()

        glutMainLoop()

    def initialize(self):
        glClearDepth(1.0)
        glClearColor(1.0, 0.0, 0.0, 0.0)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
    def initializeWrapper():
        GLWindow.get_instance().initialize()


if __name__ == '__main__':
    window = GLWindow()
