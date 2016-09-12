from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import RenderUtils

import numpy as np


class VoxelGrid():
    def __init__(self, size=(32, 32, 32), data=np.zeros((32, 32, 32))):
        self.size = size
        self.data = data

class VoxelViewer(GLWindow):

    def __init__(self, window_name="Voxel Viewer", window_size=(640, 640)):
        super(VoxelViewer, self).__init__(window_name, window_size)
        self.camera = Camera()

        self.prev_x = 0
        self.prev_y = 0
        self.initialize()

        glutMainLoop()

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

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.place()


if __name__ == "__main__":
    viewer = VoxelViewer()
