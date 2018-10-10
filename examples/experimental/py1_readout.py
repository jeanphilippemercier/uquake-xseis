import numpy as np
# import os
import matplotlib.pyplot as plt
# import glob
from importlib import reload
# import datetime
# from mayavi import mlab
# from obspy import read
# from xseis import clib
from xseis import xutil
from xseis import xplot
from xseis import xplot3d
from xseis import xobs
import h5py
plt.ion()

# hf.close()
ddir = "/home/phil/data/oyu/synthetic/"
# hf = h5py.File(ddir + 'sim_one_src.h5', 'r')
hf = h5py.File(ddir + 'sim_p5s3.h5', 'r')
# hf = h5py.File(ddir + 'sim_Vp5k.h5', 'r')
sr = hf.attrs['samplerate']
stalocs = hf['sta_locs'][:]
chanmap = hf['chan_map'][:]
src_loc = hf['src_loc'][:]
# src_time = hf['src_time'][:]
ot_true = np.argmax(np.abs(src_time))
# ot_true = np.argmax(np.abs(src_time), axis=1)[0]
rdat = hf['data'][:]
hf.close()

with np.load(ddir + "output.npz", mmap_mode='r') as npz:
	print(npz.files)
	dat = npz['sigs_preproc']
	ckeys = npz['sta_ckeys']
	ccs = npz['sigs_xcorrs']
	glims = npz['grid_lims']
	points = npz['grid_points']
	# tts = npz['ttable']
	wtt = npz['tts_to_max'].astype(int)
	gpower = npz['grid_power']
	droll = npz['sigs_rolled']
	stack = npz['sig_stack']


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
xplot3d.power_lims(gpower, glims, stalocs)

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


plt.plot(dat[0])
plt.plot(dat0[0])
plt.plot(ccs[1000])

reload(xplot)
fig = plt.figure(figsize=(6, 3))
xplot.ccf(ccs[1000], sr)

fig = plt.figure(figsize=(6, 5))
# xplot.freq(dat0[0], sr)
xplot.freq(dat[0], sr)
plt.tight_layout()


# rdat = hf['data'][:]
# plt.plot(rdat[0])
# plt.plot(dat[0])

# rdat = hf['data'][:]
dsr = sr
# dec = 4
# dsr = sr / dec
# xutil.bandpass(rdat, [80, dsr / 2], sr)
# rdat = rdat[:, ::dec]

nchan, npts = rdat.shape
wlen = 0.05 * dsr
vel = 5000.
ot = 0.175 * dsr
wins = np.arange(0, npts, wlen)

ik = np.arange(0, nchan, 10)
tts = xutil.dist2many(src_loc, stalocs) / vel * dsr
tts = tts[chanmap[ik]]

picks = ot + tts
shifts = np.arange(0, picks.shape[0], 1) * 1.2
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
xplot.sigs(rdat[ik], shifts, zorder=0, alpha=0.8)
# xplot.chan_groups(dat, groups, shifts, alpha=0.5, zorder=0)

plt.axvline(ot, linestyle='--', color='green')
ax.scatter(picks, shifts, color='green', s=50, marker='|', zorder=1, label='homo')
[plt.axvline(x, linestyle='--', color='red', alpha=0.5) for x in wins]

plt.legend()



%timeit xutil.bandpass(rdat, [80, dsr / 2], sr)
%timeit rd2 = np.ascontiguousarray(rdat[:, ::dec])
# uint32_t tt = ot + tt_ixs[ichan];
# uint32_t iblock = tt / dt;			
# uint32_t rollby = tt % dt;
# fftwf_complex *ptr_fw = fdata.row(ichan) + iblock * flen_pad;				
# process::Convolve(ptr_fw, pshift.row(rollby), &fbuf[0], flen);
# process::Accumulate(&fbuf[0], &stack[0], flen);