from renderutils import progress

import voxelizer as vox
import numpy as np
import sys
import glob

if __name__ == '__main__':
    voxels_path = glob.glob(sys.argv[1])

    print "Creating Sunflow Cubes from ", sys.argv[1]

    count = 0
    total = len(voxels_path)
    for p in voxels_path:
        vs = np.load(p)
        pts, faces = vox.volume_to_cubes(vs, threshold=0.1)
        path = p.split('.')[0]
        vox.write_cubes_obj(path+'.obj', pts, faces)
        progress(count, total)
        count += 1
    print "Done."
