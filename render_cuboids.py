from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import matplotlib.image as mpimg
import sys
import argparse
import ops

parser = argparse.ArgumentParser(description='This program generates a colored cuboid dataset.')
parser.add_argument("-w", "--width", type=int, help="Number of different widths values.", default=5)
parser.add_argument("-he", "--height", type=int, help="Number of different height values.", default=5)
parser.add_argument("-d", "--depth", type=int, help="Number of different depth values.", default=5)
parser.add_argument("-p", "--phi", type=int, help="Number of different vertical camera angles.", default=5)
parser.add_argument("-t", "--theta", type=int, help="Number of different horizontal camera angles.", default=5)
parser.add_argument("-c", "--colors", type=int, help="Number of different random color combinations.", default=5)
parser.add_argument("-s", "--size", type=int, help="Image size (it is a square image).", default=64)
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(test=False)


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
        px = self.radius * np.cos(self.theta) * np.cos(self.phi)
        py = self.radius * np.sin(self.phi)
        pz = self.radius * np.sin(self.theta) * np.cos(self.phi)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(px, py, pz, 0, 0, 0, 0, 1, 0)

MAX_COORD = 5
SCREEN_SIZE = 128
THETA_STEPS = 5
PHI_STEPS = 5
WIDTH_STEPS = 5
HEIGHT_STEPS = 5
DEPTH_STEPS = 5
COLOR_STEPS = 5
TRAIN_SET = False


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
    img_id = 0
    n_images = WIDTH_STEPS*HEIGHT_STEPS*DEPTH_STEPS*PHI_STEPS*THETA_STEPS
    img_params = np.zeros((n_images, 23))  # width + height + depth + phi + theta + 18 [6 colors, 3 floats each]
    data_config = np.array([WIDTH_STEPS, HEIGHT_STEPS, DEPTH_STEPS, PHI_STEPS, THETA_STEPS])

    img_sum = np.zeros((SCREEN_SIZE, SCREEN_SIZE, 3))

    print "Rendering dataset..."
    #for c in range(COLOR_STEPS):
    #    random_colors = np.random.rand(6, 3)
    #    cube.face_colors = random_colors
    for w in np.linspace(1.0, 6.0, WIDTH_STEPS):
        cube.width = w
        for h in np.linspace(1.0, 6.0, HEIGHT_STEPS):
            cube.height = h
            for d in np.linspace(1.0, 6.0, DEPTH_STEPS):
                cube.depth = d
                for p in np.linspace(-np.pi/2.0, np.pi/2.0, PHI_STEPS):
                    camera.phi = p
                    for t in np.linspace(0, 2*np.pi, THETA_STEPS):
                        camera.theta = t

                        random_colors = np.random.rand(6, 3)
                        cube.face_colors = random_colors

                        display()

                        buffer_data = glReadPixels(0, 0, SCREEN_SIZE, SCREEN_SIZE, GL_RGB, GL_FLOAT)
                        buffer_data = np.array(buffer_data)
                        img_sum += buffer_data
                        img_filename = "cuboid"+"_"+str(img_id)+".png"
                        if TRAIN_SET:
                            mpimg.imsave(os.path.join("data", "train", img_filename), buffer_data)
                        else:
                            mpimg.imsave(os.path.join("data", "test", img_filename), buffer_data)

                        img_params[img_id, 0:5] = np.array([w, h, d, p, t])
                        img_params[img_id, 5:23] = random_colors.flatten()
                        img_id += 1
                        ops.progress(img_id, n_images)

    img_mean = img_sum/float(img_id)
    if TRAIN_SET:
        np.save("train_params.npy", img_params)
        np.save("train_config.npy", data_config)
        np.save("train_mean.npy", img_mean)
    else:
        np.save("test_params.npy", img_params)
        np.save("test_config.npy", data_config)
        np.save("test_mean.npy", img_mean)

    print "Done."
    sys.exit()


if __name__ == '__main__':

    args = parser.parse_args()

    SCREEN_SIZE = args.size
    WIDTH_STEPS = args.width
    HEIGHT_STEPS = args.height
    DEPTH_STEPS = args.depth
    PHI_STEPS = args.phi
    THETA_STEPS = args.theta
    COLOR_STEPS = args.colors
    TRAIN_SET = args.train

    glutInit()
    glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE)
    glutCreateWindow("Color Cuboid")
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutDisplayFunc(display)
    glutIdleFunc(update)
    init()
    if not os.path.exists("data"):
        print "Data folder not found. Creating one..."
        os.makedirs("data")
        print "Done."

    if TRAIN_SET:
        if not os.path.exists(os.path.join("data", "train")):
            print "Train folder not found. Creating one..."
            os.makedirs(os.path.join("data", "train"))
            print "Done."
    else:
        if not os.path.exists(os.path.join("data", "test")):
            print "Test folder not found. Creating one..."
            os.makedirs(os.path.join("data", "test"))
            print "Done."

    glutMainLoop()
