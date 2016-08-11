from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import matplotlib.image as mpimg


class Cube:
    def __init__(self, w=3.0, h=3.0, d=3.0, fcolors=None):
        self.width = w
        self.height = h
        self.depth = d
        if fcolors is None:
            self.face_colors = [[1, 0, 0], [1, 1, 0], [1, 1, 1],
                                [0, 1, 0], [0, 0, 1], [0, 1, 1]]
        else:
            self.face_colors = fcolors

    @staticmethod
    def set_color(c):
        glColor3f(c[0], c[1], c[2])

    @staticmethod
    def set_vertex(v):
        glVertex3f(v[0], v[1], v[2])

    @staticmethod
    def draw_face(verts):
        Cube.set_vertex(verts[0])
        Cube.set_vertex(verts[1])
        Cube.set_vertex(verts[2])
        Cube.set_vertex(verts[0])
        Cube.set_vertex(verts[2])
        Cube.set_vertex(verts[3])

    def draw(self):

        glMatrixMode(GL_MODELVIEW)

        glPushMatrix()
        glScalef(self.width, self.height, self.depth)

        glBegin(GL_TRIANGLES)

        # Front face
        Cube.set_color(self.face_colors[0])
        front_face_verts = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]]
        Cube.draw_face(front_face_verts)

        # Right face
        Cube.set_color(self.face_colors[1])
        right_face_verts = [[0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
                            [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
        Cube.draw_face(right_face_verts)

        # Left face
        Cube.set_color(self.face_colors[2])
        left_face_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                           [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5]]
        Cube.draw_face(left_face_verts)

        # Back face
        Cube.set_color(self.face_colors[3])
        back_face_verts = [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                           [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        Cube.draw_face(back_face_verts)

        # Top face
        Cube.set_color(self.face_colors[4])
        top_face_verts = [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
                          [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
        Cube.draw_face(top_face_verts)

        # Bottom face
        Cube.set_color(self.face_colors[5])
        bottom_face_verts = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                             [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]
        Cube.draw_face(bottom_face_verts)

        glEnd()
        glPopMatrix()


class Camera:
    def __init__(self, t=0, p=0.5, r=5.0):
        self.theta = t
        self.phi = p
        self.radius = r

    def place(self):
        px = self.radius * np.cos(self.theta) * np.cos(self.phi);
        py = self.radius * np.sin(self.phi)
        pz = self.radius * np.sin(self.theta) * np.cos(self.phi);

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(px, py, pz, 0, 0, 0, 0, 1, 0)


MAX_COORD = 5
SCREEN_SIZE = 128
THETA_STEPS = 50
PHI_STEPS = 10


def init():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-MAX_COORD, MAX_COORD, -MAX_COORD, MAX_COORD,
            0.1, 1000.0)


cube = Cube()
camera = Camera()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    camera.place()
    cube.draw()
    glutSwapBuffers()


def update():
    for p in np.linspace(-np.pi/2.0, np.pi/2.0, PHI_STEPS):
        camera.phi = p
        for t in np.linspace(0, 2*np.pi, THETA_STEPS):
            buffer_data = glReadPixels(0, 0, SCREEN_SIZE, SCREEN_SIZE, GL_RGB, GL_FLOAT)
            buffer_data = np.array(buffer_data)
            img_filename = "cuboid"+"_"+str(t)+"_"+str(p)+".png"
            mpimg.imsave(os.path.join("data", img_filename), buffer_data)
            camera.theta = t
            display()
    print "Dataset created."


if __name__ == '__main__':
    glutInit()
    glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE)
    glutCreateWindow("Color Cuboid")
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutDisplayFunc(display)
    glutIdleFunc(update)
    init()
    if not os.path.exists("data"):
        os.makedirs("data")
    glutMainLoop()
