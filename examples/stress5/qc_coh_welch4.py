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
fband_rand = np.array([150, 200])
tt_change_percent = 0.018
noise_scale = 0.2
#####################

########################
reload(xutil)
reload(xchange)
reload(xplot)

############################

sig1, sig2 = xchange.mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
sig2 = xutil.randomize_freq_band(sig2, fband_rand, sr)
# xplot.freq(sig1, sr)
# xplot.freq(sig2, sr)

dist = 200
dvv_wlen_sec = 0.02
nwin_welch = 10
step_factor = 10
coda_start_vel = 3000.
coda_start_sec = dist / coda_start_vel
coda_end_sec = 0.8

# dvv_true = 100 * (nshift / sr)
dvv_freq_lims = np.array([60, 290])
# dvv_fband = np.array([60, 80, 270, 290])
reload(xchange)

reload(xchange)
# %%timeit
vals = xchange.dvv_phase(sig1, sig2, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, step_factor=step_factor, nwin_welch=nwin_welch)
# vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=0.2)
# dvv_out = xchange.dvv_phase_no_weights(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, step_factor=step_factor)

reload(xchange)
xchange.plot_dvv(vals, dvv_true=-tt_change_percent)

freqs = vals['freqs']
xvals = vals['xvals']

# xplot.im(vals['cohs'], norm=False, extent=[freqs[0], freqs[-1], xvals[0], xvals[-1]])

res = vals['residuals']
min_res = 2e-3
# res[res < min_res] = min_res
plt.plot(xvals, res)
xplot.axhline(min_res)

ax2 = plt.gca().twinx()
plt.plot(xvals, 1 / res, color='red')
#################
mcoh = vals['mcoh']
plt.plot(xvals, mcoh)

ax2 = plt.gca().twinx()
res = vals['residuals']
min_res = 0.01
res[res < min_res] = min_res
weights = 1 / res
plt.plot(xvals, weights, color='red')


cohs = vals['cohs']
# cohs[cohs < 0.1] = 0.1
xplot.im(cohs ** 2, norm=False, extent=[freqs[0], freqs[-1], xvals[0], xvals[-1]])
weights = 1.0 / (1.0 / (cohs ** 2) - 1.0)
# weights = 1.0 / (1.0 / (cohs) - 1.0)
# weights = 1.0 / (cohs)
# weights = 1.0 / (cohs ** 2) - 1.0
im = xplot.im(weights, norm=False, extent=[freqs[0], freqs[-1], xvals[0], xvals[-1]])
xplot.HookImageMax(im)

w = weights[0]
plt.plot(w)

w2 = w / np.sum(w)
plt.plot(w2)
# w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)

# plt.plot(xvals, np.mean(vals['cohs'], axis=1))





dvv_wlen = int(dvv_wlen_sec * sr)
iwin = [0, len(sig1)]
stepsize = dvv_wlen // step_factor

slices = xutil.build_slice_inds(iwin[0], iwin[1], dvv_wlen, stepsize=stepsize)
xv = np.mean(slices, axis=1)
# %%timeit
fdat1, freqs = xchange.fft_slices(sig1, slices, sr)
fdat2, freqs = xchange.fft_slices(sig2, slices, sr)

###################################################

from xseis2.xchange import linear_regression_zforce
# %%timeit
# %%timeit
# np.conj(fdat1)

reload(xutil)
nwin_welch = 4
# imid = nwin_welch // 2
nwin, nfreq = fdat1.shape
# igroups = xutil.build_slice_inds(0, nwin, nwin_welch, stepsize=1)
igroups = xutil.build_welch_wins(0, nwin, nwin_welch, stepsize=1)
# i0, i1 = igroups[0]

ix_fkeep = np.where((freqs > dvv_freq_lims[0]) & (freqs < dvv_freq_lims[1]))[0]

fkeep = freqs[ix_fkeep] * 2 * np.pi
# weights = np.ones(len(fkeep))

out = np.zeros((len(igroups), 2), dtype=np.float32)

cohs = []
for i, (i0, i1) in enumerate(igroups):

    # coh
    fg1 = fdat1[i0:i1]
    fg2 = fdat2[i0:i1]
    gxy = np.mean(np.conj(fg1) * fg2, axis=0)
    gxx = np.mean(np.conj(fg1) * fg1, axis=0)
    gyy = np.mean(np.conj(fg2) * fg2, axis=0)
    coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))
    weights = coh[ix_fkeep]
    cohs.append(coh)

    # dvv
    ccf = np.conj(fdat1[imid]) * fdat2[imid]
    phi = np.angle(ccf)
    phi_keep = phi[ix_fkeep]
    regress = linear_regression_zforce(fkeep, phi_keep, weights=weights)
    yint, slope, res = regress
    out[i] = [slope, res]


cohs = np.array(cohs)

xplot.im(cohs, norm=False)
########################################################






#########################


###############################
freqs = np.fft.rfftfreq(pad, 1.0 / sr)
fsr = 1.0 / (freqs[1] - freqs[0])

ix_fkeep = (np.array(freq_lims) * fsr + 0.5).astype(int)
fkeep = freqs[ix_fkeep[0]:ix_fkeep[1]] * 2 * np.pi
weights = np.ones(len(fkeep))

out = np.zeros((len(slices), 2), dtype=np.float32)

for i, sl in enumerate(slices):
    fs1 = np.fft.rfft(sig1[sl[0]:sl[1]] * taper, n=pad)
    fs2 = np.fft.rfft(sig2[sl[0]:sl[1]] * taper, n=pad)
    # ccf = np.conj(xutil.phase(fs1)) * xutil.phase(fs2)
    ccf = np.conj(fs1) * fs2
    phi = np.angle(ccf)
    phi_keep = phi[ix_fkeep[0]:ix_fkeep[1]]
    regress = linear_regression_zforce(fkeep, phi_keep, weights=weights)
    yint, slope, res = regress
    # res = np.max(np.fft.irfft(ccf))
    out[i] = [slope, res]


#################################

# %%timeit

nwin_welch = 4
nwin, nfreq = fdat1.shape
igroups = xutil.build_slice_inds(0, nwin, nwin_welch, stepsize=1)
i0, i1 = igroups[0]

cohs = []
for i, (i0, i1) in enumerate(igroups):
    fg1 = fdat1[i0:i1]
    fg2 = fdat2[i0:i1]
    gxy = np.mean(np.conj(fg1) * fg2, axis=0)
    gxx = np.mean(np.conj(fg1) * fg1, axis=0)
    gyy = np.mean(np.conj(fg2) * fg2, axis=0)
    coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))
    cohs.append(coh)

cohs = np.array(cohs)

xplot.im(cohs, norm=False)
########################################################


freqs = np.fft.rfftfreq(pad, 1.0 / sr)
fsr = 1.0 / (freqs[1] - freqs[0])

ix_fkeep = (np.array(freq_lims) * fsr + 0.5).astype(int)
fkeep = freqs[ix_fkeep[0]:ix_fkeep[1]] * 2 * np.pi
weights = np.ones(len(fkeep))

out = np.zeros((len(slices), 2), dtype=np.float32)

for i, sl in enumerate(slices):
    fs1 = np.fft.rfft(sig1[sl[0]:sl[1]] * taper, n=pad)
    fs2 = np.fft.rfft(sig2[sl[0]:sl[1]] * taper, n=pad)
    # ccf = np.conj(xutil.phase(fs1)) * xutil.phase(fs2)
    ccf = np.conj(fs1) * fs2
    phi = np.angle(ccf)
    phi_keep = phi[ix_fkeep[0]:ix_fkeep[1]]
    regress = linear_regression_zforce(fkeep, phi_keep, weights=weights)
    yint, slope, res = regress
    # res = np.max(np.fft.irfft(ccf))
    out[i] = [slope, res]




##########################################################

gxy = np.conj(fdat1) * fdat2
gxx = np.conj(fdat1) * fdat1
gyy = np.conj(fdat2) * fdat2

coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))





def f1():
    # cohs = []
    for i, (i0, i1) in enumerate(igroups):
        # print(i0, i1)
        fg1 = fdat1[i0:i1]
        fg2 = fdat2[i0:i1]
        gxy = np.mean(np.conj(fg1) * fg2, axis=0)
        gxx = np.mean(np.conj(fg1) * fg1, axis=0)
        gyy = np.mean(np.conj(fg2) * fg2, axis=0)
        coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))
        # cohs.append(coh)

    # cohs = np.array(cohs)

%timeit f1()


def f1():
    fdat1, freqs = xchange.fft_slices(sig1, slices, sr)
    fdat2, freqs = xchange.fft_slices(sig2, slices, sr)

    gxy = np.conj(fdat1) * fdat2
    gxx = np.conj(fdat1) * fdat1
    gyy = np.conj(fdat2) * fdat2


%timeit f1()


plt.plot(freqs, coh, label='coh')
plt.plot(freqs, np.abs(gxy), alpha=0.5, label='Gxy')
plt.plot(freqs, np.abs(gxx), alpha=0.5, label='Gxx')
plt.plot(freqs, np.abs(gyy), alpha=0.5, label='Gyy')
plt.legend()

##########################################


slices = xutil.build_slice_inds(iwin[0], iwin[1], dvv_wlen * 4, stepsize=stepsize)

ixl = 10
sl = slices[ixl]
w1 = sig1[sl[0]:sl[1]]
w2 = sig2[sl[0]:sl[1]]
freqs_coh, coh = coherence(w1, w2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)

plt.plot(freqs_coh, coh, label='coh')

###############################
slices = xutil.build_slice_inds(iwin[0], iwin[1], dvv_wlen * 4, stepsize=stepsize)
cohs = []
for i,sl in enumerate(slices):
    sl = slices[i]
    w1 = sig1[sl[0]:sl[1]]
    w2 = sig2[sl[0]:sl[1]]
    freqs_coh, coh = coherence(w1, w2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)
    cohs.append(coh)

cohs = np.array(cohs)
xplot.im(cohs, norm=False)




def f1():
    for i,sl in enumerate(slices):
        sl = slices[i]
        w1 = sig1[sl[0]:sl[1]]
        w2 = sig2[sl[0]:sl[1]]
        freqs_coh, coh = coherence(w1, w2, sr, nperseg=dvv_wlen, nfft=dvv_wlen * 2)
%timeit f1()


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
freqs, coh = coherence(sig1, sig2, sr, nperseg=welch_wlen, nfft=welch_nfft)
# plt.semilogy(f, Cxy)
plt.plot(freqs, coh)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.show()

#############################################
