from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import sys

from renderutils import Camera
from renderutils import GLWindow
from renderutils import RenderUtils

from voxelizer import volume_to_points

class PointCloudViewer(GLWindow):

    def __init__(self, volume_path, window_name="Mesh Viewer", window_size=(640, 640)):
        super(PointCloudViewer, self).__init__(window_name, window_size)
        self.camera = Camera()
        self.origin = np.array([0, 0, 0])

        self.threshold = 0.1
        self.volume = np.load(volume_path)
        self.point_cloud = volume_to_points(self.volume, threshold=self.threshold)

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

    def keyboard(self, k, x, y):
        if k == '.':
            self.threshold += 0.01
            np.clip(self.threshold, 0, 1)
            self.point_cloud = volume_to_points(self.volume, self.threshold)
        elif k == ',':
            self.threshold -= 0.01
            np.clip(self.threshold, 0, 1)
            self.point_cloud = volume_to_points(self.volume, self.threshold)


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

        glShadeModel(GL_SMOOTH)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.place()
        RenderUtils.color([0, 0, 1])
        RenderUtils.draw_points(self.point_cloud)
        # RenderUtils.color([0, 0, 1])
        # self.mesh.draw_normals()

if __name__ == '__main__':
    print "Use < and > keys to change the point selection threshold."
    pv = PointCloudViewer(sys.argv[1])
