import numpy as np
# import time
# from datetime import timedelta, datetime
import matplotlib.pyplot as plt
# import math
# import utm
import os
import h5py
# import itertools
# from xseis2 import xutil
from importlib import reload
from scipy import fftpack
# from scipy.fftpack import fft, ifft, fftfreq
from numpy.fft import fft, ifft, rfft, fftfreq

from scipy import interpolate
# import scipy.signal
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
from obspy.signal.regression import linear_regression


plt.ion()

siglen = 0.5
sr = 3000.0
nsamp = int(siglen * sr)
freqs = [10, 20, 450, 500]
sig = xutil.noise1d(nsamp, freqs, sr, scale=1, taplen=0.01)

tt_change_percent = 0.05

# sig = np.random.rand(nsamp)
newsig = xchange.stretch(sig, sr, tt_change_percent)
# newsig = stretch(sig, sr, tt_change_percent, len_zpad=0)

reload(xchange)
# 2 ** xchange.nextpow2(15)
# n = xchange.nextpow2(10)

time = np.arange(len(sig)) / sr

plt.plot(time, sig)
plt.plot(time, newsig)


# NAIVE - no pre-fft
print("1. NAIVE")

wlen = 50
slices = xutil.build_slice_inds(0, len(sig), wlen)
nslice = len(slices)
print("num slices", len(slices))

cclen = wlen * 2
ccs = np.zeros((len(slices), cclen)).astype(np.float32)

# start_time = time.time()
for i, sl in enumerate(slices[:]):
    win1 = sig[sl[0]:sl[1]]
    win2 = newsig[sl[0]:sl[1]]

    fs1 = np.fft.rfft(win1, n=cclen)
    fs2 = np.fft.rfft(win2, n=cclen)
    cc = np.real(np.fft.irfft(np.conj(fs1) * fs2))

    e1 = xutil.energy(win1)
    e2 = xutil.energy(win2)
    ccs[i] = cc / np.sqrt(e1 * e2)

ccs = np.roll(ccs, ccs.shape[1] // 2, axis=1)

# elapsed = time.time() - start_time
# print('elapsed: %.2f sec' % elapsed)
# print('IO: %.2f %s / sec' % (size[0] / elapsed, size[1]))

xplot.im(ccs)
tmids = slices[:, 1]
dtt = np.argmax(ccs, axis=1)
dtt / tmids

coh = np.max(ccs, axis=1)
amax = np.argmax(ccs, axis=1)
tmax = (amax - ccs.shape[1] / 2) / sr
# midtime = (slices[:, 0] + np.diff(slices)[0] / 2) / sr
midtime = (slices[:, 1]) / sr
tmax / midtime

plt.plot(time, sig / np.max(sig))
plt.plot(time, newsig / np.max(sig))
plt.scatter(slices[:, 0] / sr, coh, color='red')
# plt.scatter(slices[:, 0] / sr, tmax, color='red')


xdata = midtime
ydata = tmax
sigma = None
# sigma = 1. / weights

from scipy.optimize import curve_fit

p0 = 0.0

p, cov = curve_fit(lambda x, a: a * x, xdata, ydata, p0, sigma=sigma)
slope = p[0]
std_slope = np.sqrt(cov[0, 0])

plt.scatter(midtime, tmax)
tfit = slope * midtime
plt.plot(midtime, tfit)


# return slope, std_slope




m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())


from numpy.polynomial import polynomial as P
c, stats = P.polyfit(midtime, tmax, 1, full=True)
slope = c[0] * 100
print("tt_change: %.2f %%" % slope)

plt.scatter(midtime, tmax)
tfit = c[1] + c[0] * midtime
plt.plot(midtime, tfit)

m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())


# Get Weights
w = 1.0 / (1.0 / (coh ** 2) - 1.0)
w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
w = np.sqrt(w * np.sqrt(dcs[index_range]))
w = np.real(w)

# Frequency array:
v = np.real(freq_vec[index_range]) * 2 * np.pi

# Phase:
phi = np.angle(X)
phi[0] = 0.
phi = np.unwrap(phi)
phi = phi[index_range]

# Calculate the slope with a weighted least square linear regression
# forced through the origin
# weights for the WLS must be the variance !
m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())



######################################################
reference = sig
current = newsig
freqmin = 20
freqmax = 300
df = sr
tmin = 0.1
window_length = 0.05
step = 0.01
print(window_length * sr)

dat = xchange.mwcs(current, reference, freqmin, freqmax, df, tmin, window_length, step, smoothing_half_win=5)

# np.array([time_axis, delta_t, delta_err, delta_mcoh]).T
time_axis, delta_t, delta_err, delta_mcoh = dat.T

plt.plot(time, sig)
plt.plot(time, newsig)
plt.plot(time_axis, delta_t)

# plt.plot(time_axis, delta_t / time_axis)

plt.plot(time_axis, delta_mcoh)
# np.max(delta_mcoh)
