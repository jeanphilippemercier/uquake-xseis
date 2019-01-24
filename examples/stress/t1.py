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

plt.ion()


siglen = 0.5
sr = 3000.0
nsamp = int(siglen * sr)

freqs = [10, 20, 450, 500]
sig = xutil.noise1d(nsamp, freqs, sr, scale=1, taplen=0.01)

tt_change_percent = 2.0

# sig = np.random.rand(nsamp)
newsig = xchange.stretch(sig, sr, tt_change_percent)
# newsig = stretch(sig, sr, tt_change_percent, len_zpad=0)

# plt.plot(sig)
# plt.plot(newsig)

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
    # print(i, ' / ', nslice)
    # dat[:] = dset[:, sl[0]:sl[1]]
    # for j, ckey in enumerate(ckeys):
    # k1, k2 = ckey
    fs1 = np.fft.rfft(sig[sl[0]:sl[1]], n=cclen)
    fs2 = np.fft.rfft(newsig[sl[0]:sl[1]], n=cclen)
    # fs2 = fft(dat[k2])
    ccs[i] = np.real(np.fft.irfft(np.conj(fs1) * fs2))

# elapsed = time.time() - start_time
# print('elapsed: %.2f sec' % elapsed)
# print('IO: %.2f %s / sec' % (size[0] / elapsed, size[1]))

xplot.im(ccs)
tmids = slices[:, 1]
dtt = np.argmax(ccs, axis=1)
dtt / tmids


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
