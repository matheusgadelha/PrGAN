from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import RenderUtils

import numpy as np


class Voxel:
    def __init__(self, value=1.0, position=[0.0, 0.0, 0.0], size=1.0):
        self.value = value
        self.position = position
        self.width = size
        self.height = size
        self.depth = size
        self.face_colors = np.ones((6, 4))
        self.face_colors[:, 3] = value

    @staticmethod
    def draw_face(verts):
        RenderUtils.vertex(verts[0])
        RenderUtils.vertex(verts[1])
        RenderUtils.vertex(verts[2])
        RenderUtils.vertex(verts[0])
        RenderUtils.vertex(verts[2])
        RenderUtils.vertex(verts[3])

    def draw_cube(self):

        glMatrixMode(GL_MODELVIEW)

        glPushMatrix()
        glScalef(self.width, self.height, self.depth)

        glBegin(GL_TRIANGLES)

        # Front face
        RenderUtils.color4(self.face_colors[0])
        front_face_verts = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]
        Voxel.draw_face(front_face_verts)

        # Right face
        RenderUtils.color4(self.face_colors[1])
        right_face_verts = [[0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
                            [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
        Voxel.draw_face(right_face_verts)

        # Left face
        RenderUtils.color4(self.face_colors[2])
        left_face_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                           [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]
        Voxel.draw_face(left_face_verts)

        # Back face
        RenderUtils.color4(self.face_colors[3])
        back_face_verts = [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                           [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        Voxel.draw_face(back_face_verts)

        # Top face
        RenderUtils.color4(self.face_colors[4])
        top_face_verts = [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
                          [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        Voxel.draw_face(top_face_verts)

        # Bottom face
        RenderUtils.color4(self.face_colors[5])
        bottom_face_verts = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                             [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]
        Voxel.draw_face(bottom_face_verts)

        glEnd()
        glPopMatrix()

    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        self.draw_cube()
        glPopMatrix()


class VoxelGrid():
    def __init__(self, size=(32, 32, 32), data=np.zeros((32, 32, 32)), scale=1.0):
        self.size = size
        self.data = data
        self.scale = scale
        self.voxels = []

        wpos = np.linspace(-scale, scale, size[0], endpoint=False)
        hpos = np.linspace(-scale, scale, size[1], endpoint=False)
        dpos = np.linspace(-scale, scale, size[1], endpoint=False)

        for w_i, w in enumerate(wpos):
            for h_i, h in enumerate(hpos):
                for d_i, d in enumerate(dpos):
                    voxel = Voxel(value=data[w_i, h_i, d_i],
                                  position=[w, h, d],
                                  size=scale*2/size[0])
                    self.voxels.append(voxel)

    def draw(self):
        for v in self.voxels:
            v.draw()


class VoxelViewer(GLWindow):

    def __init__(self, window_name="Voxel Viewer", window_size=(640, 640)):
        super(VoxelViewer, self).__init__(window_name, window_size)
        self.camera = Camera()

        self.prev_x = 0
        self.prev_y = 0
        self.camera_speed = 1/100.

        data = np.zeros((16, 16, 16))
        data[:, 0, 0] = 1.0
        data[8, :, 8] = 1.0
        self.voxelgrid = VoxelGrid(size=(16, 16, 16), data=data)

        self.initialize()

        glutMainLoop()

    def initialize(self):
        MAX_COORD = 2.

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glClearColor(0.0, 0.0, 0.0, 0.0)
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
        self.voxelgrid.draw()


if __name__ == "__main__":
    viewer = VoxelViewer()
