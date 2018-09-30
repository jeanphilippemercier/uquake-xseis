import numpy as np
# import os
import matplotlib.pyplot as plt
# import glob
from importlib import reload
from xseis import xutil
from xseis import xplot
from xseis import xplot3d
from xseis import xobs
import h5py
plt.ion()

ddir = "/home/phil/data/oyu/synthetic/"
hf = h5py.File(ddir + 'sim_p5s3.h5', 'r')
sr = hf.attrs['samplerate']
stalocs = hf['sta_locs'][:]
chanmap = hf['chan_map'][:]
src_loc = hf['src_loc'][:]
ot_true = np.argmax(np.abs(src_time))
rdat = hf['data'][:]
hf.close()

with np.load(ddir + "output_new.npz", mmap_mode='r') as npz:
	print(npz.files)
	iblocks = npz['iblocks']
	bdat = npz['bdat']
	bdat2 = npz['bdat2']

	# dat = npz['sigs_preproc']
	# ckeys = npz['sta_ckeys']
	# ccs = npz['sigs_xcorrs']
	# glims = npz['grid_lims']
	# points = npz['grid_points']
	# # tts = npz['ttable']
	# wtt = npz['tts_to_max'].astype(int)
	# gpower = npz['grid_power']
	# droll = npz['sigs_rolled']
	# stack = npz['sig_stack']

wlen = 0.05 * dsr
nchan, npts = rdat.shape
wins = np.arange(0, npts, wlen)

print(np.allclose(bdat, bdat2))

xplot.sigs(bdat2[::10])
[plt.axvline(x, linestyle='--', color='red', alpha=0.5) for x in wins]


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