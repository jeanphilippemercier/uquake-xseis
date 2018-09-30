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


ddir = "/home/phil/data/oyu/synthetic/"
# MSEED = '/home/phil/data/oyu/mseed_new/20180523_185101_float.mseed'
MSEED = ddir + 'sim_dat.mseed'
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
# outgrid = np.zeros(ttP_keep.shape[1], dtype=np.float32)

outfile = ddir + 'output.npz'
xspy.fSearchWinDec2X(data, sr, stalocs, chanmap, ttP_keep,
					 outbuf, nthreads, outfile, debug_lvl)

print("power: %.2f, ix_grid: %d, ix_ot: %d" %
	 (outbuf[0] / 10000., outbuf[1], outbuf[2]))

reload(xutil)
reload(xplot3d)
lmax = xutil.imax_to_xyz_gdef(outbuf[1], gdef)

with np.load(outfile, mmap_mode='r') as npz:
	print(npz.files)
	dat = npz['sigs_preproc']
	ckeys = npz['sta_ckeys']
	ccs = npz['sigs_xcorrs']
	# glims = npz['grid_lims']
	# points = npz['grid_points']
	# tts = npz['ttable']
	wtt = npz['tts_to_max'].astype(int)
	gpower = npz['grid_power']
	droll = npz['sigs_rolled']
	stack = npz['sig_stack']


xplot3d.power(gpower, gdef, stalocs, lmax=lmax)



dw = data.copy()
xutil.whiten2D(dw, [40, 45, 300, 350], sr)


# # ttwin = (ttP_keep[:, outbuf[1]] * 2).astype(int)[chanmap]
# ttwin = (ttS_keep[:, outbuf[1]] * 2).astype(int)[chanmap]

# dr = xutil.roll_data(dw, ttwin)
# # stack = np.mean(np.abs(xutil.norm2d(dr)), axis=0)
# stack = np.mean(np.abs(dr), axis=0)

# xplot.im(dr)
# plt.axvline(outbuf[2], linestyle='--', color='red')
# plt.plot(stack / np.max(stack) * dr.shape[0] / 5, color='pink')




wloc = points[np.argmax(gpower)]
# assert(np.allclose(src, wloc))
print("correct loc?: ", np.allclose(src_loc, wloc))

ccmean = np.mean(np.max(ccs, axis=1))
print("(vmax_theor / vmax)  %.4f / %.4f " % (ccmean, np.max(gpower)))
# print("perfect tt shifts?: ", ccmean == np.max(gpower))

ot = np.argmax(stack)
ms_err = (ot_true - ot) / sr * 1000.
print("(ot_true / ot_iloc)  %d / %d  err= %.2f ms " % (ot_true, ot, ms_err))

reload(xplot3d)
# xplot3d.power_lims(gpower, glims, stalocs)
lshift = xutil.shift_locs_ot(stalocs, unshift=True)
xplot3d.power(gpower / ccmean, gdef, lshift)


plt.hist(gpower, bins=50)
plt.axvline(np.max(gpower), color='red')

reload(xutil)
reload(xplot)

groups = xutil.chan_groups(chanmap)

picks = ot + wtt
shifts = np.arange(0, picks.shape[0], 1) * 1.2
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
xplot.chan_groups(dat, groups, shifts, alpha=0.5, zorder=0)

plt.axvline(ot, linestyle='--', color='red')
plt.axvline(ot_true, linestyle='--', color='green')
ax.scatter(picks, shifts, color='red', s=30, marker='|', zorder=1, label='homo')
# ax.scatter(vmod, shifts, color='blue', s=30, marker='|', zorder=2, label='vmod')
# ax.scatter(pman, shifts, color='green', s=30, marker='|', zorder=2, label='manual')

plt.legend()


xplot.im(droll, labels=['sample (sr=3000)', 'channel'])
# plt.axvline(outbuf[2], linestyle='--', color='red')
plt.plot(stack / np.max(stack) * droll.shape[0] / 5, color='red')
plt.title('synthetic data rolled for P')



# dd = xutil.dist2many(src_loc, stalocs)[chanmap]
# isort = np.argsort(dd)
# xplot.im(rdat[isort])

# xplot.sigs(rdat[isort][:20])


mxs = np.max(ccs, axis=1)
plt.plot(mxs)
