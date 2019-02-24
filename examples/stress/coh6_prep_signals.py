import numpy as np
import matplotlib.pyplot as plt
# import math
# import utm
import os
import h5py
# import itertools
from importlib import reload
from scipy import fftpack
# from scipy.fftpack import fft, ifft, fftfreq
from numpy.fft import fft, ifft, rfft, fftfreq

from scipy import interpolate
import scipy.signal

from obspy.signal.regression import linear_regression
from obspy.signal.invsim import cosine_taper
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
from xseis2.xchange import smooth, nextpow2, getCoherence
# import scipy.fftpack
# import scipy.optimize
# import scipy.signal
# from scipy.stats import scoreatpercentile
# from scipy.fftpack.helper import next_fast_len
# from obspy.signal.regression import linear_regression


plt.ion()

wlen_sec = 0.5
sr = 3000.0
# tt_change_percent = 0.8
tt_change_percent = 0.01

reload(xchange)
reload(xchange)
nsamp = int(wlen_sec * sr)
cfreqs = [40, 50, 240, 250]
sig1 = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)
sig2 = xchange.stretch(sig1, sr, tt_change_percent)
# cfreqs = [10, 20, 800, 900]
# nscale = 1.0
# noise1 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)
# noise2 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)
#########################

# dat = np.array([sig1, sig2])

reload(xchange)
time = np.arange(len(sig1)) / sr
time = np.arange(len(sig1))
plt.plot(time, sig1)
plt.plot(time, sig2)

reload(xchange)

wlen = 50
slices = xutil.build_slice_inds(0, nsamp, wlen, stepsize=wlen // 2)
nslice = len(slices)
print("num slices", len(slices))

ccs = []
out = np.zeros((len(slices), 2))
# start_time = time.time()
for i, sl in enumerate(slices[:]):
    win1 = sig1[sl[0]:sl[1]]
    win2 = sig2[sl[0]:sl[1]]
    imax, coh, cc = xchange.measure_shift_cc(win1, win2, interp_factor=100)
    out[i] = imax, coh
    ccs.append(cc)

ccs = np.array(ccs)
# xplot.im(ccs)
# plt.plot(ccs[0] * 100)

imax, coh = out.T
print("mean coh %.3f" % np.mean(coh))

plt.plot(coh)

xv = np.mean(slices, axis=1)
# plt.scatter(xv, imax, c=coh)
# plt.scatter(xv, imax / xv * 100., c=coh)
# plt.scatter(xv / sr, imax / sr, c=coh)
imax[20] = 0.14
coh[20] = 1.0

from numpy.polynomial.polynomial import polyfit
# tmax = imax / sr
tmax = imax
c, stats = polyfit(xv, tmax, 1, full=True, w=coh)
slope = c[1] * 100
print("tt_change: %.2f %%" % slope)

# plt.scatter(midtime, tmax)
plt.scatter(xv, imax, c=coh)
tfit = c[0] + c[1] * xv
plt.plot(xv, tfit)


####################################
