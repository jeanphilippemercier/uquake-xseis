import numpy as np

# from loguru import logger
# from obspy.core from obspy.core import UTCDateTime

# from .processing_unit import ProcessingUnit

import matplotlib.pyplot as plt
import os
from importlib import reload

from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
# from xseis2.xchange import smooth, nextpow2, getCoherence
from microquake.core import read
from microquake.core.settings import settings
from glob import glob

# path = "/home/phil/data/oyu/spp_common/salv_coda.h5"

plt.ion()


sites = [station.code for station in settings.inventory.stations()]
site_locs = [station.loc for station in settings.inventory.stations()]
ldict = dict(zip(sites, site_locs))

ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
fles = np.sort(glob(os.path.join(ddir, 'dec_*.mseed')))

out = []

for i, fle in enumerate(fles):
	print(i)
	stream = read(fle)

	samplerate = stream[0].stats.sampling_rate
	lens = np.array([len(tr) for tr in stream])
	print(np.min(lens), np.max(lens))
	fixed_wlen_sec = np.max(lens) / samplerate

	rawdat, samplerate, t0 = stream.as_array(fixed_wlen_sec)
	rawdat = np.nan_to_num(rawdat)
	names = np.array([f"{tr.stats.station}.{tr.stats.channel}" for tr in stream])
	locs = np.array([ldict[n.split('.')[0]] for n in names])

	ckeys = xutil.unique_pairs(np.arange(rawdat.shape[0]))

	reload(xutil)
	ckeys = xutil.ckeys_remove_intersta(ckeys, names)
	ckeys = ckeys[:10]

	cclen = int(10 * samplerate)
	keeplag = int(2 * samplerate)
	whiten_freqs = np.array([80, 100, 220, 240])
	stack = xutil.xcorr_ckeys_stack_slices(rawdat, ckeys, cclen, keeplag, stepsize=None, whiten_freqs=whiten_freqs, sr=samplerate, onebit=True)
	out.append(stack)

out = np.array(out)
out.shape
# xplot.im(out[:, 2, :])

dd = xutil.dist_diff_ckeys(ckeys, locs)


ix = 0
dist = dd[ix]
sigs = out[:, ix, :]
sigs = sigs.reshape((-1, 2, sigs.shape[-1]))
sigs = np.mean(sigs, axis=1)
xplot.im(sigs)

sig1 = sigs[0]
sig2 = sigs[1]

# plt.plot(sig1)
# plt.plot(sig2)

hl = len(sig1) // 2
cfreqs = np.array([80, 100, 220, 240])
iwin = [hl - 1000, hl + 1000]
# iwin = [hl - 2000, hl + 2000]

wlen = 30
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))

fwins1, filt = xchange.windowed_fft(sig1, slices, samplerate, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, samplerate, cfreqs)
imax, coh = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=100).T

print("mean coh %.3f" % np.mean(coh))

xv = np.mean(slices, axis=1) - hl
# imid = np.where(np.abs(xv - hl) < 20)[0]
# imax[imid] = 0
# coh[imid] = 5
# xv[0] = 0
# imax[0] = 0
# coh[hl] = 5
reload(xchange)
outlier_val = 0.002
ik = np.where(np.abs(imax / xv) < outlier_val)[0]
# coh[:] = 1
# coh[0] = 10

# yint, slope, res = xchange.linear_regression3(xv[ik], imax[ik], coh[ik] ** 2)
yint, slope, res = xchange.linear_regression4(xv[ik], imax[ik], coh[ik] ** 2)
# yint, slope, res, ik = xchange.linear_regression2(xv, imax, coh ** 1, outlier_val)
print("tt_change: %.5f%% ss_res: %.2f " % (slope * 100, res))
tfit = yint + slope * xv

plt.scatter(xv[ik], imax[ik], c=coh[ik])
plt.colorbar()
mask = np.ones_like(xv, bool)
mask[ik] = False
plt.scatter(xv[mask], imax[mask], c='red', alpha=0.2)
# plt.scatter(xv[ik], imax[ik], c='red', alpha=0.5)
plt.plot(xv, tfit)
# plt.plot(xv, tfit)
plt.title("tt_change: %.3f%% ss_res: %.3f " % (slope * 100, res))
plt.axvline(0, linestyle='--')
alpha = 0.5
vel = 3200.
direct = dist / vel * samplerate
plt.axvline(direct, linestyle='--', color='green', alpha=alpha)
plt.axvline(-direct, linestyle='--', color='green', alpha=alpha)

####################################

XX = np.vstack((xv, np.ones_like(xv))).T
XX = xv[:, None]
# out = np.linalg.lstsq(XX[:, :-1], yv, rcond=None)



plt.plot(imax / (xv - hl))
plt.plot(imax / (xv))

################

vals = []
for ix in range(dc0.shape[0]):
	print(ix)
	sig1 = dc0[ix]
	sig2 = dc1[ix]

	hl = len(sig1) // 2
	cfreqs = [50, 60, 170, 180]
	# iwin = [-2000, 2000]
	iwin = [hl - 2000, hl + 2000]
	# wlen = 120
	wlen = 100
	stepsize = wlen // 4
	slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
	nslice = len(slices)
	# print("num slices", len(slices))

	fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
	fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
	out = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=50)

	imax, coh = out.T
	print("mean coh %.3f" % np.mean(coh))

	xv = np.mean(slices, axis=1)
	yint, slope, res, ik = xchange.linear_regression(xv, imax, coh ** 2, outlier_sd=1.5)
	print("tt_change: %.3f%% ss_res: %.2f " % (slope * 100, res))
	tfit = yint + slope * xv

	vals.append([slope * 100, res])

vals = np.array(vals)
# plt.plot(vals)
plt.plot(vals.T[0])

changes = vals.T[0]

vdict = {k: [] for k in np.unique(ckeys)}

for i, kpair in enumerate(ckeys):
	for key in kpair:
		vdict[key].append(changes[i])

avgs = []
for k, v in vdict.items():
	avgs.append(np.mean(v))

plt.plot(avgs)


fig = plt.figure(figsize=(9, 6))
x, y, z = stalocs.T
# plt.scatter(x, z, s=5, alpha=0.2)
# clrs = np.array(xplot.v2color(vmax))
# x, y, z = lkeep.T
sc = plt.scatter(x, z, s=1500, alpha=1, c=avgs, cmap='viridis')
cb = fig.colorbar(sc)
# plt.scatter(vloc[0], vloc[1], s=100, color='red')
plt.axis('equal')

plt.show()



# from xseis2 import xplot
# reload(xplot)
# xplot.stations(stalocs[:, [0, 2]], lvals=avgs, alpha=0.9)

# xplot.stations(stalocs[:, [0, 2]], ckeys=ckeys, vals=changes)
