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
###############################
# siglen = 0.1
# sr = 3000.0
# tt_change_percent = 2.0
# nsamp = int(siglen * sr)
# freqs = [40, 50, 350, 400]
# # freqs = [10, 50, 350, 400]
# # sig = xutil.noise1d(nsamp, freqs, sr, scale=1, taplen=0.01)
# # newsig = xchange.stretch(sig, sr, tt_change_percent)
# rollby = 10
# dtr = rollby / sr
# s1 = xutil.noise1d(nsamp, freqs, sr, scale=1, taplen=0.01)
# s2 = np.roll(s1, rollby)

###############################

wlen_sec = 0.1
sr = 3000.0
# tt_change_percent = 2.0
nsamp = int(wlen_sec * sr)
cfreqs = [40, 50, 200, 250]
# cfreqs = [10, 50, 350, 400]
sig = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)
# newsig = xchange.stretch(sig, sr, tt_change_percent)
cfreqs = [10, 20, 800, 900]
noise1 = xutil.noise1d(nsamp, cfreqs, sr, scale=0.2, taplen=0.01)
noise2 = xutil.noise1d(nsamp, cfreqs, sr, scale=0.2, taplen=0.01)
rollby = 10
dtr = rollby / sr
sig1 = sig
sig2 = np.roll(sig, rollby)
sig1 += noise1
sig2 += noise2

#####################################
# plt.plot(s1)
# plt.plot(s2)
window_length_samples = nsamp
padd = int(2 ** (nextpow2(2 * window_length_samples)))
# padd = next_fast_len(window_length_samples)
count = 0
tp = cosine_taper(window_length_samples, 0.85)

cci = sig1
cri = sig2

cci = scipy.signal.detrend(cci, type='linear')
cci *= tp
cri = scipy.signal.detrend(cri, type='linear')
cri *= tp

# plt.plot(cci)
# plt.plot(cri)

# padd = nsamp
# fcur1 = np.fft.rfft(cci)
# fcur2 = scipy.fftpack.fft(cci, n=padd)[:padd // 2]

fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

# plt.plot(fcur2)
# plt.plot(fref2)

smoothing_half_win = 10
# Calculate the cross-spectrum
X = fref * (fcur.conj())
if smoothing_half_win != 0:
    dcur = np.sqrt(smooth(fcur2, window='hanning',
                          half_win=smoothing_half_win))
    dref = np.sqrt(smooth(fref2, window='hanning',
                          half_win=smoothing_half_win))
    X = smooth(X, window='hanning',
               half_win=smoothing_half_win)
else:
    dcur = np.sqrt(fcur2)
    dref = np.sqrt(fref2)

dcs = np.abs(X)

# plt.plot(dcur)
# plt.plot(dref)
freqmin = 50
freqmax = 250

# Find the values the frequency range of interest
freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / sr)[:padd // 2]
index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                         freq_vec <= freqmax)).flatten()

# Get Coherence and its mean value
coh = getCoherence(dcs, dref, dcur)
mcoh = np.mean(coh[index_range])

# Get Weights
w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
w = np.sqrt(w * np.sqrt(dcs[index_range]))
w = np.real(w)

# plt.plot(w)
# plt.plot(coh)

# Frequency array:
v = np.real(freq_vec[index_range]) * 2 * np.pi

# Phase:
phi = np.angle(X)
phi[0] = 0.
phi = np.unwrap(phi)
phi = phi[index_range]

plt.plot(phi)
plt.plot(np.angle(X))

# Calculate the slope with a weighted least square linear regression
# forced through the origin
# weights for the WLS must be the variance !
m, em = linear_regression(v, phi, w)
# m, em = linear_regression(v, phi)
print(m, dtr)

delta_t.append(m)

# print phi.shape, v.shape, w.shape
e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
s2x2 = np.sum(v ** 2 * w ** 2)
sx2 = np.sum(w * v ** 2)
e = np.sqrt(e * s2x2 / sx2 ** 2)

delta_err.append(e)
delta_mcoh.append(np.real(mcoh))
time_axis.append(tmin+window_length/2.+count*step)
count += 1
