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


# path = "/home/phil/data/oyu/spp_common/salv_coda.h5"

plt.ion()

ddir = os.environ['SPP_COMMON']
hf = h5py.File(os.path.join(ddir, 'sim_coda_vel0.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel0 = hf.attrs['vp']
dat0 = hf['data'][:]
hf.close()

ddir = os.environ['SPP_COMMON']
hf = h5py.File(os.path.join(ddir, 'sim_coda_vel1.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel1 = hf.attrs['vp']
dat1 = hf['data'][:]
hf.close()
print(vel0, vel1)

plt.plot(dat0[0])
plt.plot(dat1[0])

nsta, wlen = dat0.shape
# dat = xutil.bandpass(dat, [10, 60], sr)
# dat0 = np.sign(dat0)
# dat1 = np.sign(dat1)
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
dist = dd[ix]

plt.plot(sig1)
plt.plot(sig2)

xplot.freq(dc0[6], sr)
# xplot.stations(stalocs)

###########################

tt_change_percent = 100 * (1 - (vel1 / vel0))
dvv_true = -tt_change_percent

keeplag_sec = 1.0

reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05

coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 170, 180])
whiten_freqs = fband_sig

dvv_outlier_clip = 0.1

vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
# vals['xvals'] = np.concatenate(([0], vals['xvals']))
xchange.plot_dvv(vals, dvv_true=tt_change_percent)

# sym1 = xutil.split_causals(sig1)
# plt.plot(sym1.T)

reload(xutil)
# sym1 = xutil.split_causals(sig1)[0]
# sym2 = xutil.split_causals(sig2)[0]
sym1 = xutil.symmetric(sig1)
sym2 = xutil.symmetric(sig2)
# plt.plot(sym1.T)
# reload(xchange)

reload(xchange)

vals = xchange.dvv_sym(sym1, sym2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, step_factor=10)
# vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
# vals['xvals'] = np.concatenate(([0], vals['xvals']))
xchange.plot_dvv_sym(vals, dvv_true=tt_change_percent)

# xplot.quicksave()


