from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import matplotlib.image as mpimg
import sys
import argparse
import ops


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
