
from renderutils import progress

import voxelizer as vox
import matplotlib.image as mpimg
import numpy as np
import sys
import glob
import os

def rot(t, p=0.0):
    theta = t
    phi = p

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

    return ry


def grid_coord(h, w, d):
    xl = np.linspace(-1.0, 1.0, w)
    yl = np.linspace(-1.0, 1.0, h)
    zl = np.linspace(-1.0, 1.0, d)

    xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
    g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
    return g


def transform_volume(v, t):
    height = int(v.shape[0])
    width = int(v.shape[1])
    depth = int(v.shape[2])
    grid = grid_coord(height, width, depth)

    xs = grid[0, :]
    ys = grid[1, :]
    zs = grid[2, :]

    idxs_f = np.transpose(np.vstack((xs, ys, zs)))
    idxs_f = idxs_f.dot(t)

    xs_t = (idxs_f[:, 0] + 1.0) * float(width) / 2.0
    ys_t = (idxs_f[:, 1] + 1.0) * float(height) / 2.0
    zs_t = (idxs_f[:, 2] + 1.0) * float(depth) / 2.0

    idxs = np.vstack((xs_t, ys_t, zs_t)).astype('int')
    idxs = np.clip(idxs, 0, 31)

    return np.reshape(v[idxs[0,:], idxs[1,:], idxs[2,:]], v.shape)


def project(v, tau=1):
    p = np.sum(v, axis=2)
    p = np.ones_like(p) - np.exp(-p*tau)
    return np.flipud(np.transpose(p))


if __name__ == '__main__':
    voxels_path = glob.glob(sys.argv[1])
    print "Creates dataset from volume files. Example: python create_dataset.py \"chair_volumes/*\""
    print "Creating dataset from ", sys.argv[1]

    count = 0
    total = len(voxels_path)
    for p in voxels_path:
        vs = np.load(p)
        path = p.split('.')[0]
        i = 0
        for t in np.linspace(0, 2*np.pi, 9):
            img = project(transform_volume(vs, rot(t)), 10)
            imgpath = "{}.v{}.png".format(path, i)
            mpimg.imsave(imgpath, img, cmap='gray')
            os.system("mogrify -alpha off {}".format(imgpath))
            i+=1
        progress(count, total)
        count += 1
    print "Done."
