from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from renderutils import Camera
from renderutils import GLWindow
from renderutils import Sphere
from renderutils import RenderUtils
from scipy.special import sph_harm
from mesh import Mesh
from mesh import Ray

import numpy as np
import renderutils


class SphericalHarmonicsMesh(object):
    def __init__(self, radius=1.0, resolution=50, coeffs=np.array([1])):
        self.coeffs = coeffs
        self.n_frequencies = np.sqrt(coeffs.shape[0])
        self.resolution = resolution
        self.radius = radius

        self.vertices, self.indices, self.colors = self.build_geometry()
        print self.indices
        self.mesh = Mesh(vertices=self.vertices, indices=self.indices, colors=self.colors)

    def build_geometry(self):
        theta_values = np.linspace(0, 2 * np.pi, self.resolution)
        phi_values = np.linspace(-np.pi/2., np.pi/2., self.resolution)

        vertices = []
        indices = []
        colors = []

        vertex_id = 0
        for p in range(self.resolution - 1):
            for t in range(self.resolution):
                v0, c0 = self.create_sh_vertex(theta_values[t], phi_values[p])
                vertices.append(v0)
                v0_id = vertex_id
                indices.append(v0_id)
                vertex_id += 1
                colors.append(c0)

                v1, c1 = self.create_sh_vertex(theta_values[t], phi_values[p + 1])
                vertices.append(v1)
                v1_id = vertex_id
                indices.append(v1_id)
                vertex_id += 1
                colors.append(c1)

                v2, c2 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p])
                vertices.append(v2)
                v2_id = vertex_id
                indices.append(v2_id)
                vertex_id += 1
                colors.append(c2)

                indices.append(v1_id)
                # colors.append(c1)

                v3, c3 = self.create_sh_vertex(theta_values[(t + 1) % self.resolution], phi_values[p+1])
                vertices.append(v3)
                v3_id = vertex_id
                indices.append(v3_id)
                vertex_id += 1
                colors.append(c3)

                indices.append(v2_id)
                # colors.append(c2)

        return vertices, indices, colors

    def draw(self):
        self.mesh.draw()

    def create_sh_vertex(self, theta, phi):
        r = 0.0

        for l in range(int(self.n_frequencies)):
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m
                r += self.coeffs[idx] * sph_harm(m, l, theta, phi+np.pi/2.).real

        vertex = np.array([theta, phi, r])
        return renderutils.sphere_to_cartesian(vertex),\
               renderutils.lerp(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (r/self.radius + 1)/2.)


class SphericalHarmonicsViewer(GLWindow):

    def __init__(self, window_name='Spherical Harmonics Viewer', window_size=(640, 640)):
        super(SphericalHarmonicsViewer, self).__init__(window_name, window_size)
        self.camera = Camera()
        self.origin = np.array([0, 0, 0])

        self.mesh = Mesh(os.path.join("..", "models", "cube.obj"))
        self.mesh_samples = [] # self.mesh.get_samples(1000)
        self.compute_mesh_samples(500)

        self.n_frequencies = 10
        self.sh_coeffs = np.zeros(self.n_frequencies * self.n_frequencies)
        self.compute_sh_coeffs()
        self.print_coeffs()

        self.sh = SphericalHarmonicsMesh(coeffs=self.sh_coeffs)

        self.camera_speed = 1/100.
        self.initialize()
        self.action = ""

        self.prev_x = 0
        self.prev_y = 0

        glutMainLoop()

    def compute_sh_coeffs(self):
        print "Computing SH coefficients"

        for s_i, s in enumerate(self.mesh_samples):
            for l in range(self.n_frequencies):
                for m in range(-l, l+1):
                    idx = l*(l+1) + m
                    r = np.sqrt(np.sum(np.power(s, 2)))
                    self.sh_coeffs[idx] += sph_harm(m, l,
                                                    np.arctan2(s[2], s[0]),
                                                    np.arcsin(s[1] / r)+np.pi/2.).real * self.dist_to_origin(s)
            renderutils.progress(s_i, len(self.mesh_samples))
        norm_factor = 4*np.pi / len(self.mesh_samples)
        self.sh_coeffs *= norm_factor

        print "Done."

    def print_coeffs(self):
        for l in range(self.n_frequencies):
            for m in range(-l, l + 1):
                idx = l * (l + 1) + m
                print 'l:{}, m:{}, ylm:{}'.format(l, m, self.sh_coeffs[idx])

    def dist_to_origin(self, val):
        return np.sqrt(np.sum(np.power(val - self.origin, 2)))

    # Ray tracing approach - Not good
    def compute_mesh_samples(self, count=50):
        for s in range(count):
            xi_1 = np.random.rand()
            xi_2 = np.random.rand()

            phi = 2 * np.pi * xi_2
            theta = 2 * np.arccos(np.sqrt(xi_1))

            dir = renderutils.classic_sphere_to_cartesian([phi, theta, 0.1])
            ray = Ray(self.origin, dir)
            sample = ray.intersect_mesh(self.mesh)
            if sample is not None:
                self.mesh_samples.append(sample)

    def draw_mesh_samples(self):
        glBegin(GL_POINTS)
        for s in self.mesh_samples:
            RenderUtils.vertex(s)
        glEnd()

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

        glShadeModel(GL_SMOOTH)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.place()
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self.sh.draw()
        RenderUtils.color([0, 0, 0])
        # self.draw_mesh_samples()
        # self.mesh.draw()

if __name__ == '__main__':
    window = SphericalHarmonicsViewer()