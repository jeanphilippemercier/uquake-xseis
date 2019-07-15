import os
from datetime import datetime

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
from microquake.core import read
from spp.core.settings import settings
from glob import glob

# path = "/home/phil/data/oyu/spp_common/salv_coda.h5"

plt.ion()


ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
hfname = os.path.join(ddir, '10hr_1000hz.h5')

hf = h5py.File(hfname, 'r')
names = hf['sta_names'][:].astype(str)
locs = hf['locs'][:]
sr = hf.attrs['sr']
dset = hf['data']


reload(xutil)


# pair = np.array([30, 40])
pair = np.array([33, 39])
rsig1 = dset[pair[0]]
rsig2 = dset[pair[1]]


# plt.plot(rsig1[::2])
# plt.plot(rsig1[:10000])
# plt.plot(rsig2[:10000])
xplot.freq(rsig2[:10000], sr)


l1, l2 = locs[pair]
dist = xutil.dist(l1, l2)
vel = 3200.
coda_start = int(dist / vel * sr + 0.5)
print(dist, names[pair])

whiten_freqs = np.array([80, 100, 260, 300])
# whiten_freqs = None
# stacklen = int(4 * 60 * 60 * sr)
cclen = int(10 * sr)
stacklen = int(10 * 60 * sr)
# stacklen = cclen * 10
keeplag = int(1.5 * sr)

xvals, ccs = xutil.xcorr_pair_stack_slices(rsig1, rsig2, cclen, stacklen, keeplag, stepsize=None, whiten_freqs=whiten_freqs, sr=sr, onebit=False)

xplot.im(ccs)
# xplot.im(ccs, norm=False)
# plt.axvline(0, linestyle='--')
alpha = 1.0
# vel = 3200.
# direct = dist / vel * sr
hl = ccs.shape[1] // 2
plt.axvline(hl + coda_start, linestyle='--', color='red', alpha=alpha)
plt.axvline(hl - coda_start, linestyle='--', color='red', alpha=alpha)

xplot.freq(ccs[-1], sr)

#################################################

sig1 = ccs[0]
sig2 = ccs[-1]
plt.plot(sig1)
plt.plot(sig2)


# print(f"dist {dist:.2f} m")

hl = len(sig1) // 2
cfreqs = np.array([80, 100, 250, 300])
iwin = [hl - 1000, hl + 1000]
# iwin = [hl - 2000, hl + 2000]

wlen = 30
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))

fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
imax, coh = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=100).T

print("mean coh %.3f" % np.mean(coh))

xv = np.mean(slices, axis=1) - hl

reload(xchange)
outlier_val = 0.002
ik = np.where((np.abs(xv) > coda_start) & (np.abs(imax / xv) < outlier_val))[0]

yint, slope, res = xchange.linear_regression4(xv[ik], imax[ik], coh[ik] ** 2)

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
# vel = 3200.
# direct = dist / vel * sr
plt.axvline(coda_start, linestyle='--', color='green', alpha=alpha)
plt.axvline(-coda_start, linestyle='--', color='green', alpha=alpha)



##################################


ckeys = xutil.unique_pairs(np.arange(rawdat.shape[0]))

reload(xutil)
ckeys = xutil.ckeys_remove_intersta(ckeys, names)
ckeys = ckeys[40:50]

stack = xutil.xcorr_ckeys_stack_slices(rawdat, ckeys, cclen, keeplag, stepsize=None, whiten_freqs=whiten_freqs, sr=sr, onebit=True)
out.append(stack)

out = np.array(out)
out.shape
# xplot.im(out[:, 2, :])

dd = xutil.dist_diff_ckeys(ckeys, locs)


ix = 3
print(dd[ix])
print(names[ckeys][ix])

sigs = out[:, ix, :]
# xplot.im(sigs)

sigs = sigs.reshape((-1, 2, sigs.shape[-1]))
sigs = np.mean(sigs, axis=1)

sig1 = sigs[0]
sig2 = sigs[-1]
plt.plot(sig1)
plt.plot(sig2)
xplot.im(sigs)


dist = dd[ix]
vel = 3200.
coda_start = int(dist / vel * sr + 0.5)

# print(f"dist {dist:.2f} m")


hl = len(sig1) // 2
cfreqs = np.array([80, 100, 300, 350])
iwin = [hl - 1000, hl + 1000]
# iwin = [hl - 2000, hl + 2000]

wlen = 30
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))

fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
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
ik = np.where((np.abs(xv) > coda_start) & (np.abs(imax / xv) < outlier_val))[0]

# ik = np.where(np.abs(imax / xv) < outlier_val)[0]

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
# vel = 3200.
# direct = dist / vel * sr
plt.axvline(coda_start, linestyle='--', color='green', alpha=alpha)
plt.axvline(-coda_start, linestyle='--', color='green', alpha=alpha)

####################################
