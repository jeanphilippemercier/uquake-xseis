from importlib import reload
import os
import numpy as np
import os
import time
# import h5py
# from glob import glob
from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
import matplotlib.pyplot as plt
import pickle
from scipy.signal import coherence

plt.ion()


sr = 2000.0
# dvv_wlen_sec = 0.04
# welch_wlen_sec = 0.02
cc_wlen_sec = 2.0
nsamp = int(cc_wlen_sec * sr)
fband_sig = np.array([50, 60, 290, 300])
fband_noise = np.array([5, 10, (sr / 2) - 10, sr / 2])
fband_rand = np.array([180, 200])
tt_change_percent = 0.01
noise_scale = 0.03
#####################

########################
reload(xutil)
reload(xchange)
reload(xplot)

############################

sig1, sig2 = xchange.mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
sig2 = xutil.randomize_freq_band(sig2, fband_rand, sr)

xplot.freq(sig1, sr)
xplot.freq(sig2, sr)

dvv_wlen_sec = 0.03
step_factor = 2
# welch_wlen_sec = dvv_wlen_sec / 2
# welch_wlen = int(welch_wlen_sec * sr)
# wlech_nfft = int(nsamp * 2 + 1)
# nshift = 1
# dvv_true = 100 * (nshift / sr)

reload(xchange)
# coeff = xutil.pearson_coeff(cc1, cc2)
dvv_wlen = int(dvv_wlen_sec * sr)
# coda_start = int(coda_start_sec * sr)
# coda_end = int(coda_end_sec * sr)
# iwin = [coda_start, coda_end]
# iwin = [0, coda_end]
iwin = [0, len(sig1)]
stepsize = dvv_wlen // step_factor

slices = xutil.build_slice_inds(iwin[0], iwin[1], dvv_wlen, stepsize=stepsize)
xv = np.mean(slices, axis=1)

fdat1, freqs = xchange.fft_slices(sig1, slices, sr)
fdat2, freqs = xchange.fft_slices(sig2, slices, sr)
# %timeit fdat2, freqs = xchange.fft_slices(sig2, slices, sr)
nwin, nfreq = fdat1.shape
# for i in range(nwin):
navg = 10
gxy = np.mean(np.conj(fdat1[:navg]) * fdat2[:navg], axis=0)
gxx = np.mean(np.conj(fdat1[:navg]) * fdat1[:navg], axis=0)
gyy = np.mean(np.conj(fdat2[:navg]) * fdat2[:navg], axis=0)
# gyy = np.mean(fdat2[:navg], axis=0)

# coh = np.abs(gxy) ** 2 / (np.abs(gxx) * np.abs(gyy))
coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))
# np.allclose(np.abs(gxx * gyy), (np.abs(gxx) * np.abs(gyy)))

plt.plot(freqs, coh, label='coh')
plt.plot(freqs, np.abs(gxy), alpha=0.5, label='Gxy')
plt.plot(freqs, np.abs(gxx), alpha=0.5, label='Gxx')
plt.plot(freqs, np.abs(gyy), alpha=0.5, label='Gyy')
plt.legend()

# plt.plot(freqs, abs2, alpha=0.5)
# plt.plot(freqs, abscc)

ixl = 10
sl = slices[ixl]
w1 = sig1[0:sl[1]]
w2 = sig2[0:sl[1]]
freqs_coh, coh = coherence(w1, w2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)

# freqs_coh, coh = coherence(sig1, sig2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)
# %timeit freqs_coh, coh = coherence(sig1, sig2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)

plt.plot(freqs_coh, coh)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()




# slices = xutil.build_slice_inds(iwin[0], iwin[1], dvv_wlen * 2, stepsize=stepsize)
sl = slices[5] + [-dvv_wlen, dvv_wlen]
w1 = sig1[sl[0]:sl[1]]
w2 = sig2[sl[0]:sl[1]]
freqs_coh, coh = coherence(w1, w2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)

sig1 = xutil.noise1d(nsamp, fband_sig, sr, scale=1, taplen=0.1)
# sig1 = xutil.band_noise(freq_good, sr, nsamp)
# sig2 = np.roll(sig1, 1).copy()
sig2 = sig1.copy()
# sig2 = xutil.randomize_freq_band(sig2, fband_rand, sr)
sig2 = np.roll(sig2, nshift)
xplot.freq(sig1, sr)
xplot.freq(sig2, sr)
####################################
# %timeit f, Cxy = coherence(sig1, sig2, sr, nperseg=300)
freqs, coh = coherence(sig1, sig2, sr, nperseg=welch_wlen, nfft=wlech_nfft)
# plt.semilogy(f, Cxy)
plt.plot(freqs, coh)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()

#############################################

from xseis2.xchange import linear_regression_zforce

freq_lims = fband_sig[[1, 3]]
freqmin, freqmax = freq_lims
taper_percent = 0.02
wlen_samp = nsamp
taper = xutil.taper_window(wlen_samp, taper_percent)
pad = int(2 * wlen_samp)
freqs = np.fft.rfftfreq(pad, 1.0 / sr)
ifkeep = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]

fkeep = freqs[ifkeep]
# fkeep_ang = fkeep * 2 * np.pi
# weights = np.ones(len(fkeep))
weights = coh[ifkeep] ** 2

freqs_coh, coh = coherence(sig1, sig2, sr, nperseg=welch_wlen, nfft=pad)
np.allclose(freqs, freqs_coh)

fs1 = np.fft.rfft(sig1 * taper, n=pad)
fs2 = np.fft.rfft(sig2 * taper, n=pad)
# ccf = np.conj(xutil.phase(fs1)) * xutil.phase(fs2)
ccf = np.conj(fs1) * fs2
phi = np.angle(ccf)
phi_keep = phi[ifkeep]
regress = linear_regression_zforce(fkeep, phi_keep, weights=weights)
yint, slope, res = regress

# out[i] = [slope, res]
dvv = 100 * slope / (2 * np.pi)
plt.scatter(freqs, phi, alpha=0.5)
# plt.scatter(freqs, phi, alpha=0.5, c=coh)
plt.scatter(fkeep, phi_keep, c=weights)
plt.colorbar()
xv_fit = freqs[ifkeep]
tfit = yint + slope * xv_fit
plt.plot(xv_fit, tfit, color='green', linestyle='--')
alpha = 0.3
plt.axvline(freqmin, linestyle='--', color='red', alpha=alpha)
plt.axvline(freqmax, linestyle='--', color='red', alpha=alpha)
plt.xlabel("freq (Hz)")
plt.ylabel("angle")
plt.title(f"dvv: {dvv:.4f}   dvv_true: {dvv_true:.4f}")
plt.legend()





# sig1, sig2 = xchange.mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)

# freq_good = [50, 300]
# freq_bad = [170, 200]
# = sig1[:window_length_samples].copy()
# cri += xutil.band_noise(freq_lims_bad, sr, len(cri)) * 1.0


################################
vals = xchange.dvv_phase(sig1, sig2, sr, dvv_wlen_sec, freq_lims, coda_start_sec, coda_end_sec, step_factor=step_factor)

reload(xchange)
xchange.plot_dvv(vals, dvv_true=-tt_change_percent)
# xplot.quicksave()

######################################
vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, step_factor=step_factor)

reload(xchange)
xchange.plot_dvv(vals, dvv_true=tt_change_percent)
# xplot.quicksave()


freqmin, freqmax = freq_lims
step = dvv_wlen_sec / 4
out = xchange.mwcs_msnoise(sig1, sig2, freqmin, freqmax, sr, coda_start_sec, dvv_wlen_sec, step)
time_axis, delta_t, delta_err, delta_mcoh = out

plt.scatter(time_axis, delta_t, c=delta_err)

#############################

reload(xchange)

from xseis2.xchange import smooth, getCoherence, nextpow2
import scipy
from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression as obspy_linear_regression


freqmin, freqmax = freq_lims
window_length = 0.1
df = sr
window_length_samples = np.int(window_length * df)
smoothing_half_win = 5
freq_lims_bad = [170, 200]

cci = sig1[:window_length_samples].copy()
cri = np.roll(sig1[:window_length_samples], 1).copy()
# cri += xutil.band_noise(freq_lims_bad, sr, len(cri)) * 1.0
# xplot.freq(cci, sr)
# xplot.freq(cri, sr)

padd = int(2 ** (nextpow2(window_length_samples) + 2))
tp = cosine_taper(window_length_samples, 0.85)
cci = scipy.signal.detrend(cci, type='linear')
cci *= tp
cri = scipy.signal.detrend(cri, type='linear')
cri *= tp

xplot.freq(cci, sr)
xplot.freq(cri, sr)

fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]
fcur2 = np.abs(fcur) ** 2
fref2 = np.abs(fref) ** 2

# plt.plot(fcur2)
# plt.plot(np.abs(fcur) **2)
# plt.plot(fref2)

# Calculate the cross-spectrum
X = fref * (fcur.conj())
# if smoothing_half_win != 0:
dcur = np.sqrt(smooth(fcur2, window='hanning', half_win=smoothing_half_win))
dref = np.sqrt(smooth(fref2, window='hanning', half_win=smoothing_half_win))
X = smooth(X, window='hanning', half_win=smoothing_half_win)
# dcur = np.sqrt(fcur2)
# dref = np.sqrt(fref2)

dcs = np.abs(X)

# Find the values the frequency range of interest
freq_vec = scipy.fftpack.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                         freq_vec <= freqmax))

# Get Coherence and its mean value
coh = getCoherence(dcs, dref, dcur)
mcoh = np.mean(coh[index_range])

# Get Weights
w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
w = np.sqrt(w * np.sqrt(dcs[index_range]))
w = np.real(w)
# Frequency array:
v = freq_vec[index_range] * 2 * np.pi

plt.plot(freq_vec, dcur, label='dcur = sqrt(smooth power)')
plt.plot(freq_vec, dref, label='dref = sqrt(smooth power)')
plt.plot(freq_vec, dcs, label='dcs = np.abs(smooth(cc_freq))')
plt.plot(freq_vec, coh, label='coh')
plt.plot(freq_vec[index_range], w, label='weights')
plt.legend()

# Phase:
phi = np.angle(X)
phi[0] = 0.
phi = np.unwrap(phi)
phi = phi[index_range]

# Calculate the slope with a weighted least square linear regression
# forced through the origin
# weights for the WLS must be the variance !
m, em = obspy_linear_regression(v.flatten(), phi.flatten(), w.flatten())

# delta_t.append(m)

# print phi.shape, v.shape, w.shape
e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
s2x2 = np.sum(v ** 2 * w ** 2)
sx2 = np.sum(w * v ** 2)
e = np.sqrt(e * s2x2 / sx2 ** 2)

print(m * sr)

##############


# plt.plot(vals['coh'])


# sig2r = np.roll(sig2r, 3)

from xseis2.xchange import windowed_fft, measure_shift_fwins_cc, linear_regression_zforce
from obspy.signal.regression import linear_regression as obspy_linear_regression

cc1 = sig1
cc2 = sig2
wlen_sec = dvv_wlen_sec
cfreqs = whiten_freqs
interp_factor = 100
step_factor = 4

coeff = xutil.pearson_coeff(cc1, cc2)

wlen = int(wlen_sec * sr)
coda_start = int(coda_start_sec * sr)
coda_end = int(coda_end_sec * sr)

hl = len(cc1) // 2
iwin = [hl - coda_end, hl + coda_end]

stepsize = wlen // step_factor
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
xv = np.mean(slices, axis=1) - hl

fwins1, filt = windowed_fft(cc1, slices, sr, cfreqs)
fwins2, filt = windowed_fft(cc2, slices, sr, cfreqs)
imax0, coh0 = measure_shift_fwins_cc(fwins1, fwins2, interp_factor=interp_factor).T


flims = np.array([80., 280.])
freqs = np.fft.rfftfreq(2 * wlen, 1.0 / sr)
reload(xchange)
imax1, coh1 = xchange.measure_shift_fwins_phase(fwins1, fwins2, freqs, flims).T
# imax, coh = xchange.measure_shift_fwins_phase(fwins1, fwins2, freqs, flims).T

freq_lims = np.array([80., 280.])
imax2, coh2 = xchange.measure_phase_shift_slices(cc1, cc2, slices, sr, freq_lims).T

plt.plot(imax0 / sr * -1)
plt.plot(imax1)
plt.plot(imax2)

%timeit imax2, coh2 = xchange.measure_phase_shift_slices(cc1, cc2, slices, sr, freq_lims).T


ikeep = np.arange(len(xv))
dv_old = imax / xv

iw = len(imax) - 1
fs1 = fwins1[iw]
fs2 = fwins2[iw]
ccf = np.conj(fs1) * fs2
flims = np.array([80., 280.])
freqs = np.fft.rfftfreq(2 * wlen, 1.0 / sr)
fsr = 1.0 / (freqs[1] - freqs[0])

phi = np.angle(ccf)
# plt.scatter(freqs, phi)

# flims = np.array([0, freqs[-1]])
# print(f"freq_lims: {flims}")
ixf = (flims * fsr + 0.5).astype(int)
yv = freqs[ixf[0]:ixf[1]] * 2 * np.pi
# yv = freqs[ixf[0]:ixf[1]]
phi_win = phi[ixf[0]:ixf[1]]
# plt.plot(yv, phi_win)
# v = freqs[ixf[0]:ixf[1]] * 2 * np.pi
weights = np.ones(len(yv))
regress = xchange.linear_regression_zforce(yv, phi_win, weights=weights)
yint, slope, res = regress
print(slope, dv_old[iw])

xv_fit = np.concatenate(([0], yv))
tfit = yint + slope * xv_fit
plt.plot(xv_fit, tfit)
plt.scatter(yv, phi_win)

plt.plot(imax / xv)
# dvv_percentage = -slope * 100
# print(dvv_percentage)
# slope, res = obspy_linear_regression(yv, phi_win)
# print(slope)

# timeit xchange.linear_regression_zforce(yv, phi_win, weights=weights)


plt.plot(freqs, np.cumsum(phi) / len(freqs) * 5)

plt.scatter(freqs, np.unwrap(phi))



# plt.plot(sig1r)
# plt.plot(sig2r)
#############################

# reload(xutil)
# reload(xchange)
# m, tmax = xchange.measure_phase_shift(sig1, sig2, sr)


flims = np.array([50, 300])

from xseis2.xchange import nextpow2
import scipy
from obspy.signal.invsim import cosine_taper


wlen_samp = len(sig1)
# pad = int(2 ** (nextpow2(2 * wlen_samp)))
pad = int(2 * wlen_samp)
taper = cosine_taper(wlen_samp, 0.85)

# sig1 = scipy.signal.detrend(sig1.copy(), type='linear') * taper
# sig2 = scipy.signal.detrend(sig2.copy(), type='linear') * taper

freqs = np.fft.rfftfreq(pad, 1.0 / sr)
fsr = 1.0 / (freqs[1] - freqs[0])

fs1 = np.fft.rfft(sig1, n=pad)
fs2 = np.fft.rfft(sig2, n=pad)

ccf = fs1 * np.conj(fs2)

if flims is None:
    flims = np.array([0, freqs[-1]])
print(f"freq_lims: {flims}")

ixf = (flims * fsr + 0.5).astype(int)
v = freqs[ixf[0]:ixf[1]] * 2 * np.pi

# Phase:
phi = np.angle(ccf)
phi = np.unwrap(phi)
phi = phi[ixf[0]:ixf[1]]

cc = np.fft.irfft(ccf)
cc = np.roll(cc, len(cc) // 2)
imax = np.argmax(cc) - len(cc) // 2
tmax = imax / sr

slope, res = obspy_linear_regression(v, phi)
print(slope, tmax)

yint = 0

plt.scatter(v, phi, alpha=0.2, s=5)
phi = np.angle(ccf)
plt.scatter(freqs, phi)
plt.plot(freqs, np.cumsum(phi) / len(freqs) * 5)

plt.scatter(freqs, np.unwrap(phi))


xv_fit = np.concatenate(([0], xv))
tfit = yint + slope * xv_fit
plt.plot(xv_fit, tfit, label=f'dvv_meas: {dvv_meas:.3f}')

if dvv_true is not None:
    tfit2 = yint + (dvv_true / 100) * xv_fit
    plt.plot(xv_fit, tfit2, label=f'dvv_true: {dvv_true:.3f}', color='green', linestyle='--')

plt.scatter(xv[ikeep], imax[ikeep], c=coh[ikeep])
plt.colorbar()
mask = np.ones_like(xv, bool)
mask[ikeep] = False
plt.scatter(xv[mask], imax[mask], c='red', alpha=0.2, label='ignored')

title = f"[dvv: {dvv_meas:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]"

plt.title(title)
plt.axvline(0, linestyle='--')
alpha = 0.5
# vel = 3200.
# direct = dist / vel * sr
plt.axvline(coda_start, linestyle='--', color='red', alpha=alpha)
plt.axvline(coda_end, linestyle='--', color='red', alpha=alpha)
plt.legend()


# return m, tmax































#########################
# np.corrcoef(sig1, sig2)

vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)

reload(xchange)
xchange.plot_dvv(vals, dvv_true=tt_change_percent)
xplot.quicksave()


#################################################


coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
sr = dsr

dist = 500
coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 280, 300])
fband_noise = np.array([50, 80, 280, 300])
whiten_freqs = fband_sig
tt_change_percent = 0.03
# noise_scale = 1.0
noise_scale = 0.5
dvv_outlier_clip = 1.0
# dvv_outlier_clip = None

out = []
niter = 500
for i in range(niter):
    print(f"{i} / {niter}")

    sig1, sig2 = mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
    vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    out.append(vals)

dvv = np.array([v["dvv"] for v in out])
error = [v["regress"][2] for v in out]
coeffs = [v["coeff"] for v in out]


import matplotlib.gridspec as gridspec
reload(xplot)

fig = plt.figure(figsize=(12, 8), facecolor='white')
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, :])

plt.hist(dvv, bins=70)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
plt.title(f"dvv | {niter} iters | noise_scale {noise_scale} | outlier_clip {dvv_outlier_clip} | sig {fband_sig}Hz | noise {fband_noise} Hz")
plt.xlabel("dvv measurement percent")
plt.ylabel("count")
plt.xlim(np.array([-0.05, 0.05]) + tt_change_percent)
plt.legend()

ax = fig.add_subplot(gs[1, 0])
diff = np.abs(dvv - tt_change_percent)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
# plt.scatter(diff, error, s=8, alpha=0.4)
plt.scatter(dvv, error, s=8, alpha=0.4)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("linear_fit_error")

ax = fig.add_subplot(gs[1, 1])
diff = np.abs(dvv - tt_change_percent)
# plt.scatter(diff, coeffs, s=8, alpha=0.4)
plt.scatter(dvv, coeffs, s=8, alpha=0.4)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("corr_coeff")
plt.tight_layout()
xplot.quicksave()
