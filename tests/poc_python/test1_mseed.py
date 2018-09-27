import numpy as np
# import os
import matplotlib.pyplot as plt
# import glob
from importlib import reload
# import datetime
# from mayavi import mlab
from obspy import read
from xseis2 import xspy
from xseis import xutil
from xseis import xplot
from xseis import xplot3d
from xseis import xobs

plt.ion()


# MSEED = '/home/phil/data/oyu/mseed_new/20180523_185101_float.mseed'
MSEED = '/home/phil/data/oyu/synthetic/sim_dat.mseed'
# MSEED = '/home/phil/data/oyu/OT_seismic_data/20180706112101.mseed'
# MSEED = '/home/phil/data/oyu/OT_seismic_data/20180707153222.mseed'
DIR_TTS = '/home/phil/data/oyu/NLLOC_grids/'
dsr = float(3000.)
nthreads = int(4)
debug_lvl = int(2)

ttP, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.P")
ttP = (ttP * dsr).astype(np.uint16)
ttS, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.S")
ttS = (ttS * dsr).astype(np.uint16)

st = read(MSEED)
for tr in st:
	tr.stats.station = tr.stats.station.zfill(3)
st.sort()
t0 = np.min([tr.stats.starttime for tr in st])
sr = st[0].stats.sampling_rate

reload(xobs)
data = xobs.stream_to_buffer(st, t0, npts=6000)

names = np.array([tr.stats.station for tr in st])
stakeep = np.unique(names)
stakeep_dict = dict(zip(stakeep, np.arange(len(stakeep))))
chanmap = np.array([stakeep_dict[chan] for chan in names], dtype=np.uint16)
ixkeep = np.array([ndict[k] for k in stakeep])

#########################################################

# todo: skip this copy by passing aligned ptrs
ttP_keep = ttP[ixkeep].copy()
# ttS_keep = ttS[ixkeep].copy()
stalocs = slocs[ixkeep]

outbuf = np.zeros(3, dtype=np.uint32)
outgrid = np.zeros(ttP_keep.shape[1], dtype=np.float32)

# logdir = '/home/phil/data/oyu/results/log/'
outfile = '/home/phil/data/oyu/synthetic/output.npz'
# clib.fCorrSearchDec2X(data, sr, stalocs, chanmap, ttS_keep, outbuf, outgrid, nthreads=4)
# clib.fCorrSearchDec2X(data, sr, stalocs, chanmap, tts, outbuf, outgrid, nthreads=4)
xspy.fSearchWinDec2X(data, sr, stalocs, chanmap, ttP_keep,
					 outbuf, nthreads, outfile, debug_lvl)

print("power: %.2f, ix_grid: %d, ix_ot: %d" %
	 (outbuf[0] / 10000., outbuf[1], outbuf[2]))

reload(xutil)
reload(xplot3d)
lmax = xutil.imax_to_xyz_gdef(outbuf[1], gdef)
xplot3d.power(outgrid, gdef, stalocs, lmax=lmax)


dw = data.copy()
xutil.whiten2D(dw, [40, 45, 300, 350], sr)

# ttwin = (ttP_keep[:, outbuf[1]] * 2).astype(int)[chanmap]
ttwin = (ttS_keep[:, outbuf[1]] * 2).astype(int)[chanmap]

dr = xutil.roll_data(dw, ttwin)
# stack = np.mean(np.abs(xutil.norm2d(dr)), axis=0)
stack = np.mean(np.abs(dr), axis=0)

xplot.im(dr)
plt.axvline(outbuf[2], linestyle='--', color='red')
plt.plot(stack / np.max(stack) * dr.shape[0] / 5, color='pink')
