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


plt.ion()

wlen_sec = 0.5
sr = 3000.0

ncc = 20
reload(xchange)
nsamp = int(wlen_sec * sr)
nsamp_cc = int(wlen_sec * sr * 2)

# tt_change_percent = 0.8
# tt_change_percent = 0.05
tt_changes = np.linspace(0.0, 0.05, ncc)


time = np.arange(nsamp) / sr

cfreqs = [70, 100, 350, 400]
ref_half = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.01)

ccs = np.zeros((ncc, nsamp_cc), dtype=np.float32)

for i, tt_change in enumerate(tt_changes):
    tmp = xchange.stretch(ref_half, sr, tt_change)
    cc = np.concatenate((tmp[::-1], tmp))
    ccs[i] = cc

xplot.im(ccs)

# cfreqs_noise = [30, 50, 800, 900]
# # cfreqs_noise = [300, 400, 800, 900]
# nscale = 0.02
# noise1 = xutil.noise1d(nsamp, cfreqs_noise, sr, scale=nscale, taplen=0.01)

# plt.plot(time, sig2)
# xplot.freq(sig1, sr)
# xplot.freq(noise1, sr)
# plt.plot(sig1)

#########################

reload(xutil)
reload(xchange)

wlen = 50
slices = xutil.build_slice_inds(0, nsamp_cc, wlen, stepsize=wlen // 3)
nslice = len(slices)
print("num slices", len(slices))
xv = np.mean(slices, axis=1)

# ref = ccs[0]

# sig1 = ccs[0]
# sig2 = ccs[5]
# tt_changes

results = []
fwins1, filt = xchange.windowed_fft(ccs[0], slices, sr, cfreqs)

for i, curr in enumerate(ccs[:]):

    fwins2, filt = xchange.windowed_fft(curr, slices, sr, cfreqs)
    out = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=50)
    imax, coh = out.T
    # print("mean coh %.3f" % np.mean(coh))
    yint, slope, res, ik = xchange.linear_regression(xv, imax, coh ** 2, outlier_sd=2)
    print("tt_change: %.3f%% ss_res: %.2f " % (slope * 100, res))
    results.append([slope, res])

slopes, errs = np.array(results).T

plt.plot(slopes * 100)
plt.plot(tt_changes)


# xchange.plot_tt_change(xv, imax, coh, yint, slope, res, ik)






# fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
# fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
# out = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=50)

# imax, coh = out.T
# print("mean coh %.3f" % np.mean(coh))


# yint, slope, res, ik = xchange.linear_regression(xv, imax, coh ** 2, outlier_sd=2)
# print("tt_change: %.3f%% ss_res: %.2f " % (slope * 100, res))

# xchange.plot_tt_change(xv, imax, coh, yint, slope, res, ik)
