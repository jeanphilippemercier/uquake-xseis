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
# from xseis2 import xplot

plt.ion()


def stretch(sig, sr, tt_change_percent):

    npts = len(sig)
    zpad = npts // 2
    npts_pad = npts + zpad

    psig = np.zeros(npts_pad, dtype=sig.dtype)
    psig[: -zpad] = sig

    x = np.arange(0, npts_pad) / sr
    interp = interpolate.interp1d(x, psig)

    sr_new = sr * (1 + tt_change_percent / 100.)
    xnew = np.arange(0, npts) / sr_new
    newsig = interp(xnew)

    return newsig.astype(sig.dtype)


wlen = 0.5
sr = 3000.0
nsamp = int(wlen * sr)
tt_change_percent = 2.0
# tt_change_percent = -2.0
# len_zpad = 2000

freqs = [10, 20, 450, 500]
sig = xutil.noise1d(nsamp, freqs, sr, scale=1, taplen=0.01)

# sig = np.random.rand(nsamp)
newsig = stretch(sig, sr, tt_change_percent)
# newsig = stretch(sig, sr, tt_change_percent, len_zpad=0)

plt.plot(sig)
plt.plot(newsig)


# plt.plot(times, sig)
# plt.plot(times, newsig)
# plt.show()

# x, s1, x2, s2 = stretch(sig, sr, dt)

# plt.plot(x, s1)
# plt.plot(x2, s2)


npts = len(sig)
tmin, tmax = np.array([-npts, npts]) / 2. / sr
dt_new = 1. / (sr - tt_change_percent / 100. * sr)

hlen = npts / 2 + len_zpad
x = np.arange(-hlen, hlen) / sr

psig = np.zeros(npts + len_zpad * 2)
psig[len_zpad: -len_zpad] = sig

xnew = np.arange(-npts / 2, npts / 2) * dt_new

interp = interpolate.interp1d(x, psig)
newsig = interp(xnew)

