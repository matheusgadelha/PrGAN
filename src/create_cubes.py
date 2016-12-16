from renderutils import progress

import voxelizer as vox
import numpy as np
import ops
import sys
import glob

if __name__ == '__main__':
    voxels_path = glob.glob(sys.argv[1])
    voxels_path.sort(key=ops.alphanum_key)

    print "Creating Sunflow Cubes from ", sys.argv[1]
    if len(sys.argv) < 3:
        threshold=0.1
    else:
        threshold = float(sys.argv[2])
    print "Using threshold ", threshold
    count = 0
    total = len(voxels_path)
    for p in voxels_path:
        vs = np.load(p)
        pts, faces = vox.volume_to_cubes(vs, threshold)
        vox.write_cubes_obj('cube'+str(count)+'.obj', pts, faces)
        progress(count, total)
        count += 1
    print "Done."
