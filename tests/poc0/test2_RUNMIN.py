import numpy as np
# import os
import matplotlib.pyplot as plt
# import glob
from importlib import reload
# import datetime
# from mayavi import mlab
from obspy import read
from xseis2 import xspy
from xseis2 import xflow

from xseis2 import xutil
from xseis import xplot
# from xseis import xplot3d
# from xseis import xobs

plt.ion()

ddir = "/home/phil/data/oyu/synthetic/"
mseed_file = ddir + 'sim_dat.mseed'
npz_file = ddir + 'output.npz'
tts_path = '/home/phil/data/oyu/NLLOC_grids/'

nthreads = int(4)
debug = int(2)
dsr = float(3000.)
wlen_fixed = int(1 * dsr)

ttable, stalocs, namedict, gdef = xutil.ttable_from_nll_grids(tts_path, key="OT.P")
ttable = (ttable * dsr).astype(np.uint16)
ngrid = ttable.shape[1]
tt_ptrs = np.array([row.__array_interface__['data'][0] for row in ttable])

st = read(mseed_file)
xflow.prep_stream(st, dsr)
data, t0, stations, chanmap = xflow.build_input_data(st, wlen_fixed)
ikeep = np.array([namedict[k] for k in stations])

out = xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep],
							 ngrid, nthreads, npz_file, debug)
vmax, imax, iot = out
print("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))

lmax = xutil.imax_to_xyz_gdef(imax, gdef).astype(int)
print(lmax)
true_loc = np.array([651600, 4767420, 200])
print('correct loc: ', np.allclose(lmax, true_loc))


# %time xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep], ngrid, nthreads, npz_file, debug)

# reload(xutil)
# reload(xplot3d)


# stalocs = np.ascontiguousarray(stalocs[ikeep])
# %timeit a = np.ascontiguousarray(data[::2])
