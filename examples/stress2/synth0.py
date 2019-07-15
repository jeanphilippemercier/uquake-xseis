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
# from numpy.polynomial.polynomial import polyfit


plt.ion()

wlen_sec = 1.0
sr = 1000.0
# tt_change_percent = 0.8
tt_change_percent = 0.001

reload(xchange)
nsamp = int(wlen_sec * sr)
time = np.arange(nsamp) / sr

# cfreqs = [40, 50, 240, 250]
# cfreqs = [70, 100, 350, 400]
cfreqs = np.array([80, 100, 220, 240])
sig1 = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)
sig2 = xchange.stretch(sig1, sr, tt_change_percent)

# cfreqs_noise = [30, 50, 800, 900]
# nscale = 0.02
# noise1 = xutil.noise1d(nsamp, cfreqs_noise, sr, scale=nscale, taplen=0.01)
# noise2 = xutil.noise1d(nsamp, cfreqs_noise, sr, scale=nscale, taplen=0.01)
# # plt.plot(time, sig1)

# sig1 += noise1
# sig2 += noise2
sig1 = np.concatenate((sig1[::-1][1:], sig1))
sig2 = np.concatenate((sig2[::-1][1:], sig2))
plt.plot(sig1)
plt.plot(sig2)

# plt.plot(time, noise1)

# plt.plot(time, sig2)
# xplot.freq(sig1, sr)
# xplot.freq(noise1, sr)

#########################

reload(xutil)
reload(xchange)

##########################################

samplerate = sr
dist = 280.0
vel = 3200.
coda_start = int(dist / vel * samplerate + 0.5)

# print(f"dist {dist:.2f} m")

hl = len(sig1) // 2
cfreqs = np.array([80, 100, 220, 240])
iwin = [hl - 700, hl + 700]
# iwin = [hl - 2000, hl + 2000]

wlen = 40
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))

fwins1, filt = xchange.windowed_fft(sig1, slices, samplerate, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, samplerate, cfreqs)
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
ik = np.arange(len(xv))

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
# direct = dist / vel * samplerate
plt.axvline(coda_start, linestyle='--', color='green', alpha=alpha)
plt.axvline(-coda_start, linestyle='--', color='green', alpha=alpha)







##################################
wlen = 50
slices = xutil.build_slice_inds(0, nsamp, wlen, stepsize=wlen // 3)
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
yint, slope, res, ik = xchange.linear_regression(xv, imax, coh ** 2, outlier_sd=2)
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
