from renderutils import progress

import voxelizer as vox
import numpy as np
import sys
import glob

if __name__ == '__main__':
    voxels_path = glob.glob(sys.argv[1])

    print "Creating conf OBJ from ", sys.argv[1]

    count = 0
    total = len(voxels_path)
    for p in voxels_path:
        vs = np.load(p)
        pts,conf = vox.volume_to_conf(vs)
        path = p.split('.')[0]
        vox.write_conf_obj(path+'.obj', pts, conf)
        progress(count, total)
        count += 1
    print "Done."
