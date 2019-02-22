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


nsamp = int(wlen_sec * sr)
cfreqs = [40, 50, 240, 250]
sig = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)
# newsig = xchange.stretch(sig, sr, tt_change_percent)
cfreqs = [10, 20, 800, 900]
nscale = 0.2
noise1 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)
noise2 = xutil.noise1d(nsamp, cfreqs, sr, scale=nscale, taplen=0.01)


rollby = 1
dtt_true = rollby / sr
sig1 = sig
sig2 = np.roll(sig, rollby)
sig1 += noise1
sig2 += noise2
print(f"dt/t: {dtt_true}")

reload(xchange)
flims = np.array([50., 250.])
m, dtt = xchange.phase_shift_freq(sig1, sig2, sr, flims=flims)
# %timeit xchange.phase_shift_freq(sig1, sig2, sr, flims=flims)
print(f"slope: {m} dt/t: {dtt}")
plt.plot(sig1)
plt.plot(sig2)


wlen_samp = nsamp
pad = int(2 ** (nextpow2(2 * wlen_samp)))
count = 0
taper = cosine_taper(wlen_samp, 0.85)

sig1 = scipy.signal.detrend(sig1, type='linear') * taper
sig2 = scipy.signal.detrend(sig2, type='linear') * taper

time = np.arange(0, 1, 1. / sr)

plt.plot(time, .01 * time * sr)


# plt.plot(sig1)
# plt.plot(sig2)

freqs = np.fft.rfftfreq(pad, 1.0 / sr)
# nfreq = len(freqs)
fsr = 1.0 / (freqs[1] - freqs[0])
print(len(freqs), fsr)

fs1 = np.fft.rfft(sig1, n=pad)
fs2 = np.fft.rfft(sig2, n=pad)

# ccf = np.conj(fs1) * fs2
ccf = fs1 * np.conj(fs2)
print(len(fs1), len(ccf))

# plt.plot(freqs, np.abs(fs1))
# plt.plot(freqs, np.abs(ccf))

# flims = np.array([50., 250.])
flims = np.array([100., 600.])
ixf = (flims * fsr + 0.5).astype(int)
v = freqs[ixf[0]:ixf[1]] * 2 * np.pi

# Phase:
phi = np.angle(ccf)
# phi[0] = 0.
phi = np.unwrap(phi)
phi = phi[ixf[0]:ixf[1]]

# plt.plot(phi)
# plt.plot(np.angle(ccf))

cc = np.fft.irfft(ccf)
# imax = np.argmax(cc) % len(cc)
cc = np.roll(cc, len(cc) // 2)
imax = np.argmax(cc) - len(cc) // 2
tmax = imax / sr
print(tmax, dtt)
# plt.plot(cc)

# Calculate the slope with a weighted least square linear regression
# forced through the origin
# weights for the WLS must be the variance !
# m, em = linear_regression(v, phi, w)
m, em = linear_regression(v, phi)
print(m, dtr)

plt.scatter(v, phi)
plt.plot(v, v * m, color='red')
plt.title("slope %.6f" % m)



#######################################


# padd = int(2 ** (nextpow2(2 * wlen_samp)))
padd = int(2 ** (nextpow2(2 * wlen_samp)))

fcur = scipy.fftpack.fft(sig1, n=padd)[:padd // 2]
fref = scipy.fftpack.fft(sig2, n=padd)[:padd // 2]

X = fref * (fcur.conj())

# plt.plot(dcur)
# plt.plot(dref)
freqmin = 50
freqmax = 250

freqmin, freqmax = 100, 500

# Find the values the frequency range of interest
freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / sr)[:padd // 2]
index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                         freq_vec <= freqmax)).flatten()

# Frequency array:
v = np.real(freq_vec[index_range]) * 2 * np.pi

# Phase:
phi = np.angle(X)
phi[0] = 0.
phi = np.unwrap(phi)
phi = phi[index_range]

plt.plot(phi)
plt.plot(np.angle(X))

m, em = linear_regression(v, phi)
print(m, dtr)

plt.scatter(v, phi)
plt.plot(v, v * m, color='red')
plt.title("slope %.6f" % m)




