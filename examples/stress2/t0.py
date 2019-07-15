import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload
from scipy import fftpack
from numpy.fft import fft, ifft, rfft, fftfreq

from scipy import interpolate
import scipy.signal

from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
# from xseis2.xchange import smooth, nextpow2, getCoherence
from numpy.polynomial.polynomial import polyfit
from microquake.core import read


# path = "/home/phil/data/oyu/spp_common/salv_coda.h5"

plt.ion()

ddir = os.environ['SPP_COMMON']

st = read('/home/phil/data/oyu/spp_common/continuous/test.mseed')


hf = h5py.File(os.path.join(ddir, 'sim_coda.h5'), 'r')
# hf = h5py.File(os.path.join(ddir, 'salv_coda2.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel0 = hf.attrs['vp']
dat0 = hf['data'][:]
hf.close()
print(sr)

ddir = os.environ['SPP_COMMON']
# hf = h5py.File(os.path.join(ddir, 'sim_coda2.h5'), 'r')
# hf = h5py.File(os.path.join(ddir, 'sim_coda3.h5'), 'r')
hf = h5py.File(os.path.join(ddir, 'sim_coda4.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel1 = hf.attrs['vp']
dat1 = hf['data'][:]
hf.close()
print(sr)

plt.plot(dat0[0])
plt.plot(dat1[0])

nsta, wlen = dat0.shape
# dat = xutil.bandpass(dat, [10, 60], sr)
# dat = np.sign(dat)
# xplot.freq(dat[0], sr)
reload(xutil)
ckeys = xutil.unique_pairs(np.arange(nsta))
dd = xutil.dist_diff_ckeys(ckeys, stalocs)
# ckeys = ckeys[dd > 500]
dc0 = xutil.xcorr_ckeys(dat0, ckeys)
dc1 = xutil.xcorr_ckeys(dat1, ckeys)
# dc0 = xutil.causal(dc0)
# dc1 = xutil.causal(dc1)

ix = 100
sig1 = dc0[ix]
sig2 = dc1[ix]

plt.plot(sig1)
plt.plot(sig2)

xplot.freq(dc0[6], sr)
# xplot.stations(stalocs)


reload(xutil)
reload(xchange)

hl = len(sig1) // 2
cfreqs = [50, 60, 170, 180]
# iwin = [-2000, 2000]
iwin = [hl - 2000, hl + 2000]
wlen = 100
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))

fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
out = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=50)

imax, coh = out.T
print("mean coh %.3f" % np.mean(coh))

xv = np.mean(slices, axis=1)
# xv[0] = 0
# imax[0] = 0
# coh[0] = 5
yint, slope, res, ik = xchange.linear_regression(xv, imax, coh ** 2, outlier_sd=1.5)
print("tt_change: %.3f%% ss_res: %.2f " % (slope * 100, res))
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
