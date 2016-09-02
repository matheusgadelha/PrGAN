from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow

import numpy as np
#import matplotlib.image as mpimg
#import sys
#import argparse
#import ops

class SphericalHarmonicsViewer(GLWindow):

    def __init__(self, window_name='Spherical Harmonics Viewer', window_size=(640, 640)):
        super(SphericalHarmonicsViewer, self).__init__(window_name, window_size)

    def initialize(self):
        glClearColor(1.0, 1.0, 0.0, 0.0)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

if __name__ == '__main__':
    window = SphericalHarmonicsViewer()