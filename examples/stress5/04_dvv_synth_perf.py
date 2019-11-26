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


# path = "/home/phil/data/oyu/spp_common/salv_coda.h5"

plt.ion()

ddir = os.environ['SPP_COMMON']
hf = h5py.File(os.path.join(ddir, 'sim_coda_vel0.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr_raw = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel0 = hf.attrs['vp']
dat0 = hf['data'][:]
hf.close()

ddir = os.environ['SPP_COMMON']
hf = h5py.File(os.path.join(ddir, 'sim_coda_vel1.h5'), 'r')
print(list(hf.keys()))
stalocs = hf['sta_locs'][:]
srclocs = hf['src_loc'][:]
sr_raw = hf.attrs['samplerate']
polys = hf['polys'][:][:, :, [0, 2]]
vel1 = hf.attrs['vp']
dat1 = hf['data'][:]
hf.close()
print(vel0, vel1)

# plt.plot(dat0[0])
# plt.plot(dat1[0])

nsta, wlen = dat0.shape
# dat = xutil.bandpass(dat, [10, 60], sr)
# xplot.freq(dat[0], sr)
reload(xutil)
ckeys = xutil.unique_pairs(np.arange(nsta))
dd = xutil.dist_diff_ckeys(ckeys, stalocs)
# ckeys = ckeys[dd > 500]
dc0 = xutil.xcorr_ckeys(dat0, ckeys)
dc1 = xutil.xcorr_ckeys(dat1, ckeys)

decf = 2
sr = sr_raw / decf

dc0 = dc0[:, ::decf]
dc1 = dc1[:, ::decf]

nkeep = int(1 * sr)
dc0 = xutil.keeplag(dc0, nkeep)
dc1 = xutil.keeplag(dc1, nkeep)

ix = 100
sig1 = dc0[ix]
sig2 = dc1[ix]
dist = dd[ix]

plt.plot(sig1)
plt.plot(sig2)

nsamp = len(sig1)

xplot.freq(dc0[6], sr)
# xplot.stations(stalocs)

###########################

tt_change_percent = 100 * (1 - (vel1 / vel0))
dvv_true = -tt_change_percent

# keeplag_sec = 1.0
reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.5
dvv_wlen_sec = 0.05
step_factor = 4

coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 170, 180])
whiten_freqs = fband_sig

dvv_outlier_clip = 0.1
##############################################

# plt.plot(sig1)
# plt.plot(sig2)

# vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, step_factor=step_factor)
%timeit vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, step_factor=step_factor)
# vals['xvals'] = np.concatenate(([0], vals['xvals']))
xchange.plot_dvv(vals, dvv_true=tt_change_percent)

################################################

# keeplag_sec = 1.0
reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.5
dvv_wlen_sec = 0.05
step_factor = 4

coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 170, 180])
whiten_freqs = fband_sig

dvv_outlier_clip = 0.1


def measure_shift_fwins_cc(fwins1, fwins2, interp_factor=1):

    nwin, nfreq = fwins1.shape
    pad = nfreq * 2 - 1

    out = np.zeros((nwin, 2), dtype=np.float32)

    for i, (w1, w2) in enumerate(zip(fwins1, fwins2)):
        ccf = np.conj(w1) * w2
        # cc = np.fft.irfft(ccf, n=pad * interp_factor)
        cc = np.fft.irfft(ccf, n=pad)
        cc = np.roll(cc, len(cc) // 2)
        imax = (np.argmax(cc) - len(cc) // 2) / interp_factor
        out[i] = [imax, np.max(cc) * interp_factor]

    return out


from xseis2.xchange import windowed_fft, linear_regression_zforce
# from xseis2.xchange import windowed_fft, measure_shift_fwins_cc, linear_regression_zforce

cc1 = sig1
cc2 = sig2
wlen_sec = dvv_wlen_sec
cfreqs = whiten_freqs
interp_factor = 100


coeff = xutil.pearson_coeff(cc1, cc2)

wlen = int(wlen_sec * sr)
coda_start = int(coda_start_sec * sr)
coda_end = int(coda_end_sec * sr)

hl = len(cc1) // 2
iwin = [hl - coda_end, hl + coda_end]

stepsize = wlen // step_factor
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)

fwins1, filt = windowed_fft(cc1, slices, sr, cfreqs)
fwins2, filt = windowed_fft(cc2, slices, sr, cfreqs)
%timeit imax, coh = measure_shift_fwins_cc(fwins1, fwins2, interp_factor=interp_factor).T
imax, coh = measure_shift_fwins_cc(fwins1, fwins2, interp_factor=interp_factor).T

xv = np.mean(slices, axis=1) - hl
ikeep = np.arange(len(xv))

is_coda = np.abs(xv) > coda_start
n_outlier = 0

if dvv_outlier_clip is not None:
    is_outlier = np.abs(imax / xv) < (dvv_outlier_clip / 100)
    ikeep = np.where((is_coda) & (is_outlier))[0]
    n_outlier = int(100 * (1 - (np.sum(is_outlier) / len(is_outlier))))
    # print(f"non-outlier: {np.sum(is_outlier) / len(is_outlier) * 100:.2f}%")
else:
    ikeep = np.where((is_coda))[0]

regress = linear_regression_zforce(xv[ikeep], imax[ikeep], coh[ikeep] ** 2)

    # regress = [0, 0, 0]

yint, slope, res = regress
dvv_percentage = -slope * 100

print(f"[dvv: {dvv_percentage:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]")

out = {"dvv": float(dvv_percentage), "regress": regress, "xvals": xv, "imax": imax, "coh": coh, "coda_win": [coda_start, coda_end], "ikeep": ikeep, "coeff": float(coeff), "n_outlier": float(n_outlier)}

vals = out
xchange.plot_dvv(vals, dvv_true=tt_change_percent)














##############################
# sym1 = xutil.split_causals(sig1)
# plt.plot(sym1.T)

reload(xutil)
# sym1 = xutil.split_causals(sig1)[0]
# sym2 = xutil.split_causals(sig2)[0]
sym1 = xutil.symmetric(sig1)
sym2 = xutil.symmetric(sig2)
# plt.plot(sym1.T)
# reload(xchange)

reload(xchange)

vals = xchange.dvv_sym(sym1, sym2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, step_factor=10)
# vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
# vals['xvals'] = np.concatenate(([0], vals['xvals']))
xchange.plot_dvv_sym(vals, dvv_true=tt_change_percent)

# xplot.quicksave()


