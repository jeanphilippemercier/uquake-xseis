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
from xseis2 import xplot
from xseis2 import xplot3d
# from xseis import xobs

plt.ion()

ddir = "/home/phil/data/oyu/synthetic/"
mseed_file = ddir + 'sim_dat.mseed'
# mseed_file = ddir + 'sim_dat2.mseed'
# mseed_file = '/home/phil/data/oyu/sim_dat2.mseed'
npz_file = ddir + 'output.npz'
tts_path = '/home/phil/data/oyu/NLLOC_grids/'


nthreads = int(4)
debug = int(2)
dsr = float(3000.)
wlen_sec = 1.0
# wlen_fixed = int(1 * dsr)

ttable, stalocs, namedict, gdef = xutil.ttable_from_nll_grids(tts_path, key="OT.P")
# ttable, stalocs, namedict, gdef = xutil.ttable_from_nll_grids(tts_path, key="OT.S")
ttable = (ttable * dsr).astype(np.uint16)
ngrid = ttable.shape[1]
tt_ptrs = np.array([row.__array_interface__['data'][0] for row in ttable])

reload(xutil)
reload(xflow)
st = read(mseed_file)
xflow.prep_stream(st)
data, t0, stations, chanmap = xflow.build_input_data(st, wlen_sec, dsr)
ikeep = np.array([namedict[k] for k in stations])

out = xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep],
							 ngrid, nthreads, npz_file, debug)
vmax, imax, iot = out
print("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))

lmax = xutil.imax_to_xyz_gdef(imax, gdef).astype(int)
print(lmax)
true_loc = np.array([651600, 4767420, 200])
print('correct loc: ', np.allclose(lmax, true_loc))

####################################################################################

with np.load(npz_file, mmap_mode='r') as npz:
	print(npz.files)
	dat = npz['dat_preproc']
	ckeys = npz['sta_pairs']
	ccs = npz['dat_cc']
	wtt = npz['tts_src'].astype(int)
	gpower = npz['grid_power']
	droll = npz['dat_rolled']
	stack = npz['dat_stack']


# xplot3d.power(gpower, gdef, stalocs, lmax=lmax)
xplot3d.power(gpower, gdef, stalocs[ikeep], labels=stations)


xplot.im(droll, labels=['sample (samplerate=%d)' % dsr, 'channel'])
plt.plot(stack / np.max(stack) * droll.shape[0] / 5, color='red')
plt.title('synthetic data rolled for P')

# ttwin = (ttP_keep[:, out[1]]).astype(int)
# print(np.allclose(ttwin, wtt))


groups = xutil.chan_groups(chanmap)

picks = ot + wtt
shifts = np.arange(0, picks.shape[0], 1) * 1.2
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
xplot.chan_groups(dat, groups, shifts, alpha=0.5, zorder=0)

plt.axvline(ot, linestyle='--', color='red')
ax.scatter(picks, shifts, color='red', s=30, marker='|', zorder=1, label='homo')
plt.legend()


mxs = np.max(ccs, axis=1)
plt.plot(mxs)



# %time xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep], ngrid, nthreads, npz_file, debug)
# reload(xutil)
# reload(xplot3d)
# stalocs = np.ascontiguousarray(stalocs[ikeep])
# %timeit a = np.ascontiguousarray(data[::2])

# dist = 3050.0
# minvel = 3100.
# time = dist / minvel



