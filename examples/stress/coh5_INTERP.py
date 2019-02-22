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

wlen_sec = 0.1
sr = 3000.0
# tt_change_percent = 2.0

reload(xchange)
nsamp = int(wlen_sec * sr)
cfreqs = [40, 50, 240, 250]
sig = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)
# newsig = xchange.stretch(sig, sr, tt_change_percent)
cfreqs = [10, 20, 800, 900]
nscale = 0.1
noise1 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)
noise2 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)


def measure_shift_cc(sig1, sig2, interp_factor=1):

    wlen_samp = len(sig1)
    # pad = int(2 ** (nextpow2(2 * wlen_samp)))
    pad = int(2 * wlen_samp)
    taper = cosine_taper(wlen_samp, 0.85)
    sig1 = scipy.signal.detrend(sig1, type='linear') * taper
    sig2 = scipy.signal.detrend(sig2, type='linear') * taper

    # freqs = np.fft.rfftfreq(pad, 1.0 / sr)
    # fsr = 1.0 / (freqs[1] - freqs[0])

    fs1 = np.fft.rfft(sig1, n=pad)
    fs2 = np.fft.rfft(sig2, n=pad)

    ccf = fs1 * np.conj(fs2)

    cc = np.fft.irfft(ccf, n=pad * interp_factor)
    cc = np.roll(cc, len(cc) // 2)
    imax = (np.argmax(cc) - len(cc) // 2) / interp_factor
    # tmax = imax / sr

    return cc, imax

rollby = 5
rollby = 0.01
dtt_true = rollby
sig1 = sig.copy()
sig2 = xchange.apply_phase_shift(sig, rollby)
# sig2 = np.roll(sig1, rollby)
# sig1 += noise1
# sig2 += noise2

cc, imax = measure_shift_cc(sig1, sig2, interp_factor=100)
print(imax)

# plt.plot(cc)

plt.plot(sig1)
plt.plot(sig2)


# plt.plot(sig3)

# t1 = np.linspace(0, nsamp / sr, nsamp)
# t2 = np.linspace(0, nsamp / sr, nsamp * 2)
# plt.plot(t1, sig1)
# plt.plot(t1, sig2)
# plt.plot(t2, sig3)


# reload(xchange)
# flims = np.array([40., 200.])
# m, dtt = xchange.phase_shift_freq(sig1, sig2, sr, flims=flims)
# # %timeit xchange.phase_shift_freq(sig1, sig2, sr, flims=flims)

# # print(f"dt/t_true: {dtt_true}")
# # print(f"slope: {m * 100}\ndt/t_xcorr: {dtt * 100}")

# reload(xchange)
# freqmin, freqmax = flims

# out = xchange.mwcs_msnoise_single(sig1, sig2, freqmin, freqmax, sr, smoothing_half_win=5)
# # print(f"slope_msnoise: {out[0] * 100}")
# print(f"true dt/t: {dtt_true}")
# # print(f"slope dt/t: {m * 100}")
# # print(f"xcorr dt/t: {dtt * 100}")
# print(f"slope dt/t: {m * sr}")
# print(f"xcorr dt/t: {dtt * sr}")
# print(f"msnoise dt/t: {out[0] * sr}")

# plt.plot(np.real(out[2]))
