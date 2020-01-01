import numpy as np
from scipy import interpolate
import logging

import scipy.fftpack
import scipy.optimize
import scipy.signal
# from scipy.stats import scoreatpercentile
# from scipy.fftpack.helper import next_fast_len
from xseis2 import xutil
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from loguru import logger
from numba import njit


def ot_keep_sites():

    sites = np.array([18,  20,  22,  24,  26,  28,  30,  32,  41,  44,  46,  48,  50,
                52,  54,  56,  58,  63,  65,  67,  69,  71,  79,  81,  83,  86,
                88,  90, 104, 106, 108, 110, 112, 115, 117, 119, 121, 126, 131])
    return sites


def ot_best_pairs():
    return np.array(['.110.X_.41.X', '.110.Y_.41.X', '.110.Y_.41.Y', '.110.Z_.41.X',
       '.110.Z_.41.Y', '.121.X_.41.X', '.121.X_.41.Y', '.121.Y_.41.X',
       '.121.Y_.41.Y', '.121.Z_.41.X', '.121.Z_.41.Y', '.24.X_.41.X',
       '.24.X_.41.Y', '.24.Y_.41.X', '.24.Y_.41.Y', '.117.X_.119.X', '.117.X_.119.Y', '.117.Z_.119.Y', '.117.Z_.119.Z', '.50.X_.52.Y', '.50.X_.52.Z', '.50.Y_.52.Y', '.50.Y_.52.Z'],
      dtype='<U13')


def ot_bad_chans():
     return np.array(['.131.X', '.131.Y', '.131.Z', '.56.Y', '.67.Y', '.83.X', '.86.X',
       '.86.Y', '.86.Z', '.88.X', '.88.Y', '.88.Z', '.90.X', '.90.Z'],
      dtype='<U6')


def signal_quality(sig):

    npts = len(sig)
    nans = len(np.where(sig == np.nan)[0])
    zeros = len(np.where(sig == 0)[0])
    pos_ratio = 100 * len(np.where(sig > 0)[0]) / npts
    zeros_per = zeros / npts * 100
    nans_per = nans / npts * 100

    print(f"npts: {npts} | zeros: {zeros} ({zeros_per:.2f}%) | nans: {nans} ({nans_per:.2f}%) | npositive {pos_ratio:.3f}%")

    return {'zeros': zeros_per, 'nans': nans_per, 'npos': pos_ratio}


def cc_noise_level(dat):
    c1 = np.mean(dat ** 2, axis=0)
    c2 = np.mean(dat, axis=0) ** 2
    noise = np.sqrt((c1 - c2) / (dat.shape[0] - 1))

    return noise


def assess_snr(xcorrs_dat, nsmooth):
    stack = np.mean(xcorrs_dat, axis=0)
    signal = xutil.smooth(xutil.envelope(stack), nsmooth)

    noise = cc_noise_level(xcorrs_dat)
    noise = xutil.smooth(noise, nsmooth)

    snr = signal / noise

    return snr, signal, noise


def mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale):

    keeplag = nsamp // 2
    sig1 = xutil.noise1d(keeplag, fband_sig, sr, scale=1, taplen=0)
    sig2 = stretch(sig1, sr, tt_change_percent)

    # sig1 = np.concatenate((sig1[::-1][1:], sig1))
    # sig2 = np.concatenate((sig2[::-1][1:], sig2))
    sig1 = np.concatenate((sig1[::-1], sig1))
    sig2 = np.concatenate((sig2[::-1], sig2))
    noise1 = xutil.noise1d(nsamp, fband_noise, sr, scale=noise_scale, taplen=0)
    noise2 = xutil.noise1d(nsamp, fband_noise, sr, scale=noise_scale, taplen=0)
    sig1 += noise1
    sig2 += noise2
    return sig1, sig2


def stretch_and_mirror(half_in, sr, tt_change_percent, fband_noise, noise_scale):

    # keeplag = nsamp // 2
    # sig1 = xutil.noise1d(keeplag, fband_sig, sr, scale=1, taplen=0)
    half = stretch(half_in, sr, tt_change_percent)
    full = np.concatenate((half[::-1], half))
    if noise_scale > 0:
        full += xutil.noise1d(len(full), fband_noise, sr, scale=noise_scale, taplen=0)
    return full, half


def linear_regression_zforce(xv, yv, weights):
    out = np.linalg.lstsq(xv[:, None] * weights[:, None], yv * weights, rcond=None)
    # print(out)
    slope = out[0][0]
    yint = 0
    residual = out[1][0] / len(xv)
    # residual = out[1][0]

    return yint, slope, residual


def dvv_phase(cc1, cc2, sr, wlen_sec, freq_lims, coda_start_sec, coda_end_sec, step_factor=4, nwin_welch=4):

    cclen = len(cc1)
    assert(cclen == len(cc2))

    times = xutil.xcorr_lagtimes(cclen, sr)
    icoda = np.where((np.abs(times) < coda_end_sec) & (np.abs(times) > coda_start_sec))

    coeff = xutil.pearson_coeff(cc1[icoda], cc2[icoda])

    wlen = int(wlen_sec * sr)
    coda_start = int(coda_start_sec * sr)
    coda_end = int(coda_end_sec * sr)
    hl = len(cc1) // 2
    iwin = [hl - coda_end, hl + coda_end]

    stepsize = wlen // step_factor
    slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
    fdat1, freqs = fft_slices(cc1, slices, sr)
    fdat2, _ = fft_slices(cc2, slices, sr)
    nwin, nfreq = fdat1.shape

    ix_fkeep = np.where((freqs > freq_lims[0]) & (freqs < freq_lims[1]))[0]
    fkeep = freqs[ix_fkeep] * 2 * np.pi

    welch_groups = xutil.build_welch_wins(0, nwin, nwin_welch, stepsize=1)

    n_measure = len(welch_groups)
    out = np.zeros((n_measure, 2), dtype=np.float32)
    # breakpoint()

    cohs = []
    mcoh = []
    for i, (i0, imid, i1) in enumerate(welch_groups):
        # print(i)
        # coh
        fg1 = fdat1[i0:i1]
        fg2 = fdat2[i0:i1]
        gxy = np.mean(np.conj(fg1) * fg2, axis=0)
        gxx = np.mean(np.conj(fg1) * fg1, axis=0)
        gyy = np.mean(np.conj(fg2) * fg2, axis=0)
        coh = np.abs(gxy) ** 2 / (np.abs(gxx * gyy))
        mcoh.append(np.mean(coh[ix_fkeep]))
        # weights = coh[ix_fkeep] ** 2
        weights = 1.0 / (1.0 / (coh[ix_fkeep] ** 2) - 0.9999)
        weights /= np.sum(weights)
        cohs.append(coh)
        # cohs.append(weights)
        # if i == 98:
            # breakpoint()

        # dvv
        ccf = np.conj(fdat1[imid]) * fdat2[imid]
        phi = np.angle(ccf)
        phi_keep = phi[ix_fkeep]
        regress = linear_regression_zforce(fkeep, phi_keep, weights=weights)
        yint, slope, res = regress
        # res = np.sum((phi_keep - slope * fkeep) ** 2) / (np.size(fkeep) - 1)
        res = np.sum((np.sqrt(weights) * (phi_keep - slope * fkeep)) ** 2)
        out[i] = [slope, res]

        # m=slope, phi=phi_keep, v=fkeep 2pi, w=weights
        # e = np.sum(np.abs(phi_keep - slope * fkeep)) / (np.size(fkeep) - 1)
        # s2x2 = np.sum(fkeep ** 2 * weights ** 2)
        # sx2 = np.sum(weights * fkeep ** 2)
        # e = np.sqrt(e * s2x2 / sx2 ** 2)

        # out[i] = [slope, res / len(fkeep)]
        # out[i] = [slope, res]

    cohs = np.array(cohs)
    mcoh = np.array(mcoh)
    dtt, errors = out.T
    # coh = np.mean(cohs, axis=1)
    res = errors.copy()
    min_res = 0
    res[res < min_res] = min_res
    weights = 1 / res
    # weights = np.ones(len(errors))

    imax = dtt * sr

    # coh = 1 + 1 / (errors + 0.1)
    # coh = 0.5 + 1 / (errors + 0.05)
    # coh = errors
    # w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        # w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
    # coh = np.ones(len(imax))

    xv = np.mean(slices[welch_groups[:, 1]], axis=1) - hl

    ikeep = np.arange(len(xv))
    is_coda = np.abs(xv) > coda_start
    n_outlier = 0
    ikeep = np.where((is_coda))[0]

    if n_outlier < 90:
        regress = linear_regression_zforce(xv[ikeep], imax[ikeep], weights[ikeep])
    else:
        regress = [0, 0, 0]

    yint, slope, res = regress
    dvv_percentage = -slope * 100

    print(f"[dvv: {dvv_percentage:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]")

    out = {"dvv": float(dvv_percentage), "regress": regress, "xvals": xv, "imax": imax, "coh": weights, "coda_win": [coda_start, coda_end], "ikeep": ikeep, "coeff": float(coeff), "n_outlier": float(n_outlier), "cohs": cohs, "residuals": errors, "freqs": freqs, 'sr': sr, 'mcoh': mcoh}

    return out


def dvv_phase_no_weights(cc1, cc2, sr, wlen_sec, freq_lims, coda_start_sec, coda_end_sec, step_factor=4):

    assert(len(cc1) == len(cc2))
    coeff = xutil.pearson_coeff(cc1, cc2)

    wlen = int(wlen_sec * sr)
    coda_start = int(coda_start_sec * sr)
    coda_end = int(coda_end_sec * sr)

    hl = len(cc1) // 2
    iwin = [hl - coda_end, hl + coda_end]

    stepsize = wlen // step_factor
    slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
    dtt, errors = measure_phase_shift_slices(cc1, cc2, slices, sr, freq_lims).T

    imax = dtt * sr

    xv = np.mean(slices, axis=1) - hl
    ikeep = np.arange(len(xv))

    is_coda = np.abs(xv) > coda_start
    n_outlier = 0
    ikeep = np.where((is_coda))[0]

    # xv_time = xv / sr
    # coh = np.ones(len(errors))
    coh = 1 / (errors + 0.5)
    # coh = errors ** 2

    # if dvv_outlier_clip is not None:
    #     is_outlier = np.abs(imax / xv) < (dvv_outlier_clip / 100)
    #     ikeep = np.where((is_coda) & (is_outlier))[0]
    #     n_outlier = int(100 * (1 - (np.sum(is_outlier) / len(is_outlier))))
    #     # print(f"non-outlier: {np.sum(is_outlier) / len(is_outlier) * 100:.2f}%")
    # else:
    #     ikeep = np.where((is_coda))[0]

    if n_outlier < 90:
        regress = linear_regression_zforce(xv[ikeep], imax[ikeep], coh[ikeep])
        # regress = linear_regression_zforce(xv_time[ikeep], dtt[ikeep], coh[ikeep] ** 2)
    else:
        regress = [0, 0, 0]

    yint, slope, res = regress
    dvv_percentage = -slope * 100

    print(f"[dvv: {dvv_percentage:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]")

    out = {"dvv": float(dvv_percentage), "regress": regress, "xvals": xv, "imax": imax, "coh": coh, "coda_win": [coda_start, coda_end], "ikeep": ikeep, "coeff": float(coeff), "n_outlier": float(n_outlier), "errors": errors}

    return out


def dvv(cc1, cc2, sr, wlen_sec, cfreqs, coda_start_sec, coda_end_sec, interp_factor=100, dvv_outlier_clip=None, step_factor=4):

    assert(len(cc1) == len(cc2))

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

    if n_outlier < 90:
        regress = linear_regression_zforce(xv[ikeep], imax[ikeep], coh[ikeep] ** 2)
    else:
        regress = [0, 0, 0]

    yint, slope, res = regress
    dvv_percentage = -slope * 100

    print(f"[dvv: {dvv_percentage:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]")

    out = {"dvv": float(dvv_percentage), "regress": regress, "xvals": xv, "imax": imax, "coh": coh, "coda_win": [coda_start, coda_end], "ikeep": ikeep, "coeff": float(coeff), "n_outlier": float(n_outlier)}

    return out


def dvv_sym(cc1, cc2, sr, wlen_sec, cfreqs, coda_start_sec, coda_end_sec, interp_factor=100, dvv_outlier_clip=None, step_factor=4):

    assert(len(cc1) == len(cc2))

    coeff = xutil.pearson_coeff(cc1, cc2)

    wlen = int(wlen_sec * sr)
    coda_start = int(coda_start_sec * sr)
    coda_end = int(coda_end_sec * sr)

    iwin = [coda_start, coda_end]
    # iwin = [0, coda_end]

    stepsize = wlen // step_factor
    slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
    xv = np.mean(slices, axis=1)

    fwins1, filt = windowed_fft(cc1, slices, sr, cfreqs)
    fwins2, filt = windowed_fft(cc2, slices, sr, cfreqs)
    imax, coh = measure_shift_fwins_cc(fwins1, fwins2, interp_factor=interp_factor).T

    ikeep = np.arange(len(xv))

    is_coda = xv > coda_start
    n_outlier = 0

    if dvv_outlier_clip is not None:
        is_outlier = np.abs(imax / xv) < (dvv_outlier_clip / 100)
        ikeep = np.where((is_coda) & (is_outlier))[0]
        n_outlier = int(100 * (1 - (np.sum(is_outlier) / len(is_outlier))))
        # print(f"non-outlier: {np.sum(is_outlier) / len(is_outlier) * 100:.2f}%")
    else:
        ikeep = np.where((is_coda))[0]

    if n_outlier < 90:
        regress = linear_regression_zforce(xv[ikeep], imax[ikeep], coh[ikeep] ** 2)
    else:
        regress = [0, 0, 0]

    yint, slope, res = regress
    dvv_percentage = -slope * 100

    print(f"[dvv: {dvv_percentage:.4f}%] [corr_coeff: {coeff:.2f}] [outliers: {n_outlier}%] [res_fit: {res:.4f}]")

    out = {"dvv": float(dvv_percentage), "regress": regress, "xvals": xv, "imax": imax, "coh": coh, "coda_win": [coda_start, coda_end], "ikeep": ikeep, "coeff": float(coeff), "n_outlier": float(n_outlier)}

    return out


def fft_slices(sig, slices, sr, taper_percent=0.2):

    # print("num slices", len(slices))
    wlen_samp = slices[0][1] - slices[0][0]
    pad = int(2 * wlen_samp)
    taper = xutil.taper_window(wlen_samp, taper_percent)
    freqs = np.fft.rfftfreq(pad, 1.0 / sr)
    nfreq = len(freqs)

    fdat = np.zeros((len(slices), nfreq), dtype=np.complex64)

    for i, sl in enumerate(slices):
        fsig = np.fft.rfft(sig[sl[0]:sl[1]] * taper, n=pad)
        fdat[i] = fsig

    return fdat, freqs


def windowed_fft(sig, slices, sr, corner_freqs, taper_percent=0.2):

    # print("num slices", len(slices))
    wlen_samp = slices[0][1] - slices[0][0]
    pad = int(2 * wlen_samp)
    taper = xutil.taper_window(wlen_samp, taper_percent)
    freqs = np.fft.rfftfreq(pad, 1.0 / sr)
    nfreq = len(freqs)

#     fband = [5, 7, 18, 22]
    filt = xutil.freq_window(corner_freqs, pad, sr)
    # norm = 1.0 / xutil.energy(filt) * 2

    fdat = np.zeros((len(slices), nfreq), dtype=np.complex64)

    for i, sl in enumerate(slices):
        fsig = np.fft.rfft(sig[sl[0]:sl[1]] * taper, n=pad)
        fdat[i] = xutil.norm_energy_freq(filt * xutil.phase(fsig))
        # fdat[i] = xutil.norm_energy_freq(fsig)

    return fdat, filt


def measure_shift_fwins_cc(fwins1, fwins2, interp_factor=1):

    nwin, nfreq = fwins1.shape
    pad = nfreq * 2 - 1

    out = np.zeros((nwin, 2), dtype=np.float32)

    for i, (w1, w2) in enumerate(zip(fwins1, fwins2)):
        ccf = np.conj(w1) * w2
        cc = np.fft.irfft(ccf, n=pad * interp_factor)
        cc = np.roll(cc, len(cc) // 2)
        imax = (np.argmax(cc) - len(cc) // 2) / interp_factor
        out[i] = [imax, np.max(cc) * interp_factor]

    return out


def measure_phase_shift_slices(sig1, sig2, slices, sr, freq_lims, taper_percent=0.2):

    # print("num slices", len(slices))
    wlen_samp = slices[0][1] - slices[0][0]
    # nslices = len(slices)
    taper = xutil.taper_window(wlen_samp, taper_percent)

    pad = int(2 * wlen_samp)
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

    return out


def measure_shift_fwins_phase(fwins1, fwins2, freqs, flims):

    nwin, nfreq = fwins1.shape
    fsr = 1.0 / (freqs[1] - freqs[0])
    ixf = (flims * fsr + 0.5).astype(int)
    yv = freqs[ixf[0]:ixf[1]] * 2 * np.pi
    weights = np.ones(len(yv))

    out = np.zeros((nwin, 2), dtype=np.float32)

    for i, (w1, w2) in enumerate(zip(fwins1, fwins2)):
        ccf = np.conj(w1) * w2
        phi = np.angle(ccf)
        phi_win = phi[ixf[0]:ixf[1]]
        regress = linear_regression_zforce(yv, phi_win, weights=weights)
        yint, slope, res = regress
        out[i] = [slope, res]

    return out


# def measure_shift_fwins_phase(fwins1, fwins2, freqs, flim):

#     nwin, nfreq = fwins1.shape
#     fsr = 1.0 / (freqs[1] - freqs[0])
#     flims = np.array([80., 280.])
#     ixf = (flims * fsr + 0.5).astype(int)
#     yv = freqs[ixf[0]:ixf[1]] * 2 * np.pi
#     weights = np.ones(len(yv))

#     out = np.zeros((nwin, 2), dtype=np.float32)

#     for i, (w1, w2) in enumerate(zip(fwins1, fwins2)):
#         ccf = np.conj(w1) * w2
#         phi = np.angle(ccf)
#         phi_win = phi[ixf[0]:ixf[1]]
#         regress = linear_regression_zforce(yv, phi_win, weights=weights)
#         yint, slope, res = regress
#         out[i] = [slope, res]

#     return out


def plot_dvv_sym(vals, dvv_true=None):
    import matplotlib.pyplot as plt

    yint, slope, res = vals['regress']
    xv = vals['xvals']
    coh = vals['coh']
    imax = vals['imax']
    ikeep = vals['ikeep']
    coeff = vals['coeff']
    n_outlier = vals['n_outlier']
    coda_start, coda_end = vals['coda_win']
    dvv_meas = vals['dvv']

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


def plot_dvv_inds(vals, dvv_true=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8))

    yint, slope, res = vals['regress']
    xv = vals['xvals']
    coh = vals['coh']
    imax = vals['imax']
    ikeep = vals['ikeep']
    coeff = vals['coeff']
    n_outlier = vals['n_outlier']
    coda_start, coda_end = vals['coda_win']
    dvv_meas = vals['dvv']

    tfit = yint + slope * xv
    plt.plot(xv, tfit, label=f'dvv_meas: {dvv_meas:.3f}')

    if dvv_true is not None:
        tfit2 = yint + (dvv_true / 100) * xv
        plt.plot(xv, tfit2, label=f'dvv_true: {dvv_true:.3f}', color='green', linestyle='--')

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
    plt.axvline(-coda_start, linestyle='--', color='red', alpha=alpha)
    plt.legend()
    plt.tight_layout()


def plot_dvv(vals, dvv_true=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 8))

    yint, slope, res = vals['regress']
    slope *= 100
    coh = vals['coh']
    ikeep = vals['ikeep']
    coeff = vals['coeff']
    n_outlier = vals['n_outlier']
    dvv_meas = vals['dvv']
    sr = vals['sr']

    imax = vals['imax'] / sr * 100
    coda_start, coda_end = np.array(vals['coda_win']) / sr
    xv = vals['xvals'] / sr

    tfit = yint + slope * xv
    plt.plot(xv, tfit, label=f'dvv_meas: {dvv_meas:.3f}')

    if dvv_true is not None:
        # tfit2 = yint + (dvv_true / 100) * xv
        tfit2 = yint + dvv_true * xv
        plt.plot(xv, tfit2, label=f'dvv_true: {dvv_true:.3f}', color='green', linestyle='--')

    plt.scatter(xv[ikeep], imax[ikeep], c=coh[ikeep], alpha=0.8)
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
    plt.axvline(-coda_start, linestyle='--', color='red', alpha=alpha)
    plt.xlabel("Lag time (seconds)")
    plt.ylabel("dvv (%)")
    plt.legend()
    plt.tight_layout()


def stream_decompose(st, wlen_sec=None, starttime=None):

    if starttime is None:
        starttime = np.min([tr.stats.starttime for tr in st])
    sr = st[0].stats.sampling_rate
    nchan = len(st)

    if wlen_sec is not None:
        npts_fix = int(wlen_sec * sr)
    else:
        npts_fix = int(np.max([len(tr.data) for tr in st]))

    data = np.zeros((nchan, npts_fix), dtype=np.float32)
    chan_names = []

    for i, tr in enumerate(st):
        i0 = int((tr.stats.starttime - starttime) * sr + 0.5)
        sig = tr.data
        slen = min(len(sig), npts_fix - i0)
        data[i, i0: i0 + slen] = sig[:slen]
        chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

    return data, sr, starttime, np.array(chan_names)


def xcorr_ckeys_stack_slices(rawdat, sr, ckeys, cc_wlen_sec, keeplag_sec, stepsize_sec=None, whiten_freqs=None, onebit=True, pos_ratio_range=[45, 55]):

    ncc = len(ckeys)
    cclen = int(cc_wlen_sec * sr)
    keeplag = int(keeplag_sec * sr)
    stepsize = int(stepsize_sec * sr)

    padlen = int(cc_wlen_sec * sr * 2)
    nfreq = int(padlen // 2 + 1)
    # whiten_freqs = np.array(whiten_freqs)

    logger.info(f'ncc: {ncc}, cclen: {cc_wlen_sec}s, keeplag: {keeplag_sec}s, stepsize: {stepsize_sec} s')
    # print(cclen, padlen, keeplag, stepsize)

    nchan, nsamp = rawdat.shape
    slices = xutil.build_slice_inds(0, nsamp, cclen, stepsize=stepsize)
    nslices = len(slices)

    whiten_win = None
    if whiten_freqs is not None:
        whiten_freqs = np.array(whiten_freqs)
        whiten_win = xutil.freq_window(whiten_freqs, padlen, sr)

    logger.info(f'Computing {ncc} xcorrs for {nslices} slices of {cc_wlen_sec}s each')

    cc_stack_freq = np.zeros((ncc, nfreq), dtype=np.complex64)
    fdat = np.zeros((nchan, nfreq), dtype=np.complex64)
    # quality = np.zeros((nchan, nslices), dtype=np.float32)
    nstack = np.zeros(ncc, dtype=np.float32)
    stack_flag = np.zeros(nchan, dtype=bool)

    for i, sl in enumerate(slices):
        # print(f"stacking slice {i} / {nslices}")
        fdat.fill(0)

        for isig in range(nchan):
            sig = rawdat[isig, sl[0]:sl[1]].copy()
            pos_ratio = 100 * len(np.where(sig > 0)[0]) / len(sig)
            # quality[isig][i] = pos_ratio

            if pos_ratio < pos_ratio_range[0] or pos_ratio > pos_ratio_range[1]:
                # print(f"chan {isig}: pos_ratio {pos_ratio:.1f} not in range, setting to zero")
                stack_flag[isig] = 0
                continue

            stack_flag[isig] = 1

            fsig = np.fft.rfft(sig, n=padlen)

            if whiten_win is not None:
                fsig = whiten_win * xutil.phase(fsig)
            else:
                fsig /= np.sqrt(xutil.energy_freq(fsig))

            if onebit is True:
                sig = np.sign(np.fft.irfft(fsig))
                fsig = np.fft.rfft(sig)
                fsig /= np.sqrt(xutil.energy_freq(fsig))

            fdat[isig] = fsig

        cc_ikeep = np.where(np.sum(stack_flag[ckeys], axis=1) == 2)[0]
        nstack[cc_ikeep] += 1
        xcorr_stack_freq(fdat, ckeys, cc_stack_freq, cc_ikeep)

    keeplag = int(keeplag)
    cc_stack = np.zeros((ncc, keeplag * 2), dtype=np.float32)

    for i in range(ncc):
        cc = np.fft.irfft(cc_stack_freq[i]) / nslices
        cc_stack[i, :keeplag] = cc[-keeplag:]
        cc_stack[i, keeplag:] = cc[:keeplag]

    nstack = nstack * 100 / nslices

    return cc_stack, nstack


@njit
def xcorr_stack_freq(sigs_freq, cc_keys, cc_stack_freq, cc_ikeep):

    for ixc in cc_ikeep:
        k1, k2 = cc_keys[ixc]
        cc_stack_freq[ixc] += np.conj(sigs_freq[k1]) * sigs_freq[k2]


def plot_tt_change(xv, imax, coh, yint, slope, res, ik):

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


def measure_shift_cc(sig1, sig2, interp_factor=1, taper_percent=0.2):

    wlen_samp = len(sig1)
    pad = int(2 * wlen_samp)
    taper = xutil.taper_window(wlen_samp, taper_percent)
    # freqs = np.fft.rfftfreq(pad, 1.0 / sr)
    # nfreq = len(freqs)
    # fsr = 1.0 / (freqs[1] - freqs[0])

    fs1 = np.fft.rfft(sig1 * taper, n=pad)
    fs2 = np.fft.rfft(sig2 * taper, n=pad)

    fs1 /= np.sqrt(xutil.energy_freq(fs1))
    fs2 /= np.sqrt(xutil.energy_freq(fs2))

    ccf = np.conj(fs1) * fs2

    cc = np.fft.irfft(ccf, n=pad * interp_factor) * interp_factor
    cc = np.roll(cc, len(cc) // 2)
    imax = (np.argmax(cc) - len(cc) // 2) / interp_factor
    # tmax = imax / sr

    return imax, np.max(cc), cc
    # return cc, imax


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


def stretch_sim_ccs(cclen_sec, sr, tt_changes, cfreqs):
    wlen_sec = cclen_sec / 2
    ncc = len(tt_changes)
    nsamp = int(wlen_sec * sr)
    nsamp_cc = int(wlen_sec * sr * 2)
    ref_half = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.0)
    ccs = np.zeros((ncc, nsamp_cc), dtype=np.float32)

    for i, tt_change in enumerate(tt_changes):
        tmp = stretch(ref_half, sr, tt_change)
        cc = np.concatenate((tmp[::-1], tmp))
        ccs[i] = cc

    return ccs

# def nextpow2(val):
#     buf = math.ceil(math.log(val) / math.log(2))
#     return int(math.pow(2, buf))


def nextpow2(x):

    return np.ceil(np.log2(np.abs(x)))


def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
    # TODO: docsting
    window_len = 2 * half_win + 1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype(x.dtype)
    else:
        w = scipy.signal.hanning(window_len).astype(x.dtype)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]


def getCoherence(dcs, ds1, ds2):
    # TODO: docsting
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh


def apply_phase_shift(sig, nshift):
    fsig = np.fft.rfft(sig)
    nfreq = len(fsig)
    fstep = 0.5 / (nfreq - 1)
    factor = -nshift * 2.0 * np.pi * fstep
    vec = np.zeros(nfreq, dtype=np.complex64)
    inds = np.arange(nfreq)
    vec.real = np.cos(inds * factor)
    vec.imag = np.sin(inds * factor)
    return np.fft.irfft(vec * fsig)


def measure_phase_shift(sig1, sig2, sr, flims=None):

    from obspy.signal.invsim import cosine_taper
    from obspy.signal.regression import linear_regression as obspy_linear_regression

    wlen_samp = len(sig1)
    pad = int(2 ** (nextpow2(2 * wlen_samp)))
    taper = cosine_taper(wlen_samp, 0.85)

    sig1 = scipy.signal.detrend(sig1.copy(), type='linear') * taper
    sig2 = scipy.signal.detrend(sig2.copy(), type='linear') * taper

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

    m, em = obspy_linear_regression(v, phi)

    return m, tmax


def mwcs_msnoise(current, reference, freqmin, freqmax, df, tmin, window_length, step,
         smoothing_half_win=5):
    """
    :type current: :class:`numpy.ndarray`
    :param current: The "Current" timeseries
    :type reference: :class:`numpy.ndarray`
    :param reference: The "Reference" timeseries
    :type freqmin: float
    :param freqmin: The lower frequency bound to compute the dephasing (in Hz)
    :type freqmax: float
    :param freqmax: The higher frequency bound to compute the dephasing (in Hz)
    :type df: float
    :param df: The sampling rate of the input timeseries (in Hz)
    :type tmin: float
    :param tmin: The leftmost time lag (used to compute the "time lags array")
    :type window_length: float
    :param window_length: The moving window length (in seconds)
    :type step: float
    :param step: The step to jump for the moving window (in seconds)
    :type smoothing_half_win: int
    :param smoothing_half_win: If different from 0, defines the half length of
        the smoothing hanning window.

    :rtype: :class:`numpy.ndarray`
    :returns: [time_axis,delta_t,delta_err,delta_mcoh]. time_axis contains the  central times of the windows. The three other columns contain dt, error and
    mean coherence for each window.
    """

    from obspy.signal.invsim import cosine_taper
    from obspy.signal.regression import linear_regression as obspy_linear_regression

    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = np.int(window_length * df)

    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    # padd = next_fast_len(window_length_samples)
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)
    minind = 0
    maxind = window_length_samples
    while maxind <= len(current):
        cci = current[minind:(minind + window_length_samples)]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = reference[minind:(minind + window_length_samples)]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(step*df)
        maxind += int(step*df)

        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

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
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin
        # weights for the WLS must be the variance !
        m, em = obspy_linear_regression(v.flatten(), phi.flatten(), w.flatten())

        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        # m=slope, phi=phi_keep, v=fkeep 2pi, w=weights
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+window_length/2.+count*step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(current) + step*df:
        logging.warning("The last window was too small, but was computed")

    return np.array([time_axis, delta_t, delta_err, delta_mcoh])


def mwcs_msnoise_single(cci, cri, freqmin, freqmax, df, smoothing_half_win=5):

    window_length_samples = len(cci)

    # window_length_samples = np.int(window_length * df)

    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    # padd = next_fast_len(window_length_samples)
    # count = 0
    tp = cosine_taper(window_length_samples, 0.85)
    # minind = 0
    # maxind = window_length_samples
    # while maxind <= len(current):
    #     cci = current[minind:(minind + window_length_samples)]
    cci = scipy.signal.detrend(cci, type='linear')
    cci *= tp

    # cri = reference[minind:(minind + window_length_samples)]
    cri = scipy.signal.detrend(cri, type='linear')
    cri *= tp

    # minind += int(step * df)
    # maxind += int(step * df)

    fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
    fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

    fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
    fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

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
    v = np.real(freq_vec[index_range]) * 2 * np.pi

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

    # delta_err.append(e)
    # delta_mcoh.append(np.real(mcoh))
    # time_axis.append(tmin+window_length/2.+count*step)
    # count += 1

    # return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T
    return m, e, coh



# def xcorr_ckeys_stack_slices2(rawdat, sr, ckeys, cc_wlen_sec, keeplag_sec, stepsize_sec=None, whiten_freqs=None, onebit=True, random_amp=1e-9):

#     ncc = len(ckeys)
#     cclen = int(cc_wlen_sec * sr)
#     keeplag = int(keeplag_sec * sr)
#     stepsize = int(stepsize_sec * sr)

#     padlen = int(cc_wlen_sec * sr * 2)
#     nfreq = int(padlen // 2 + 1)
#     whiten_freqs = np.array(whiten_freqs)

#     logger.info(f'ncc: {ncc}, cclen: {cc_wlen_sec}s, keeplag: {keeplag_sec}s, stepsize: {stepsize_sec} s')
#     # print(cclen, padlen, keeplag, stepsize)

#     nchan, nsamp = rawdat.shape
#     slices = xutil.build_slice_inds(0, nsamp, cclen, stepsize=stepsize)

#     whiten_win = None
#     if whiten_freqs is not None:
#         whiten_win = xutil.freq_window(whiten_freqs, padlen, sr)

#     logger.info(f'Computing {ncc} xcorrs for {len(slices)} slices of {cc_wlen_sec}s each')

#     cc_stack_freq = np.zeros((ncc, nfreq), dtype=np.complex64)

#     for i, sl in enumerate(slices):
#         print(f"stacking slice {i} / {len(slices)}")
#         dat = rawdat[:, sl[0]:sl[1]].copy()

#         if onebit is True:
#             dat = xutil.bandpass(dat, whiten_freqs[[0, -1]], sr)
#             dat = np.sign(dat)

#         fdat = np.fft.rfft(dat, axis=1, n=padlen)

#         if whiten_win is not None:
#             for irow in range(fdat.shape[0]):
#                 fdat[irow] = whiten_win * xutil.phase(fdat[irow])
#         else:
#             for irow in range(fdat.shape[0]):
#                 fdat[irow] /= np.sqrt(xutil.energy_freq(fdat[irow]))

#         xcorr_stack_freq(fdat, ckeys, cc_stack_freq)

#     keeplag = int(keeplag)
#     stack = np.zeros((ncc, keeplag * 2), dtype=np.float32)

#     for i in range(ncc):
#         cc = np.fft.irfft(cc_stack_freq[i]) / len(slices)
#         stack[i, :keeplag] = cc[-keeplag:]
#         stack[i, keeplag:] = cc[:keeplag]

#     return stack


# def xcorr_stack_slices_gen(datgen, chans, cclen, sr_raw, sr_dec, keeplag, whiten_freqs, onebit=True):

#     ckeys = xutil.unique_pairs(np.arange(len(chans)))
#     ckeys = xutil.ckeys_remove_intersta(ckeys, chans)

#     ncc = len(ckeys)
#     padlen = int(cclen * sr_dec * 2)
#     nfreq = int(padlen // 2 + 1)
#     whiten_freqs = np.array(whiten_freqs)

#     dec_factor = None
#     if not np.allclose(sr_raw, sr_dec):
#         dec_factor = int(sr_raw / sr_dec)

#     whiten_win = None
#     if whiten_freqs is not None:
#         whiten_win = xutil.freq_window(whiten_freqs, padlen, sr_dec)

#     cc_stack_freq = np.zeros((ncc, nfreq), dtype=np.complex64)

#     nstack = 0
#     for dat in datgen:

#         dat = xutil.bandpass(dat, whiten_freqs[[0, -1]], sr_raw)

#         if dec_factor is not None:
#             dat = dat[:, ::dec_factor]

#         if onebit is True:
#             dat = np.sign(dat)

#         fdat = np.fft.rfft(dat, axis=1, n=padlen)

#         for irow in range(fdat.shape[0]):
#             fdat[irow] = whiten_win * xutil.phase(fdat[irow])

#         for j, ckey in enumerate(ckeys):
#             k1, k2 = ckey
#             # stack[j] = np.fft.irfft(np.conj(fdat[k1]) * fdat[k2])
#             cc_stack_freq[j] += np.conj(fdat[k1]) * fdat[k2]

#         nstack += 1

#     nlag = int(keeplag * sr_dec)
#     stack = np.zeros((ncc, nlag * 2), dtype=np.float32)

#     for i in range(ncc):
#         cc = np.real(np.fft.irfft(cc_stack_freq[i])) / nstack
#         stack[i, :nlag] = cc[-nlag:]
#         stack[i, nlag:] = cc[:nlag]

#     return stack, ckeys


# def xcorr_stack_slices2(hstream, starttime, endtime, chans, cclen, keeplag, whiten_freqs, onebit=True):

#     # slices = xutil.build_slice_inds(i0, i1, cclen, stepsize=stepsize)
#     sr = hstream.samplerate
#     ckeys = xutil.unique_pairs(np.arange(len(chans)))
#     ckeys = xutil.ckeys_remove_intersta(ckeys, chans)

#     ncc = len(ckeys)
#     padlen = int(cclen * sr * 2)
#     nfreq = int(padlen // 2 + 1)

#     whiten_win = None
#     if whiten_freqs is not None:
#         whiten_win = xutil.freq_window(whiten_freqs, padlen, sr)

#     cc_stack_freq = np.zeros((ncc, nfreq), dtype=np.complex64)
#     cclen_td = timedelta(seconds=cclen)
#     t0 = starttime
#     t1 = starttime + cclen_td
#     nstack = 0

#     # sliding window of cclen until endtime
#     while (t1 < endtime - cclen_td):
#         print(t0, t1)
#         dat = hstream.query(chans, t0, t1)

#         if onebit is True:
#             dat = xutil.bandpass(dat, whiten_freqs[[0, -1]], sr)
#             dat = np.sign(dat)

#         fdat = np.fft.rfft(dat, axis=1, n=padlen)

#         for irow in range(fdat.shape[0]):
#             fdat[irow] = whiten_win * xutil.phase(fdat[irow])

#         for j, ckey in enumerate(ckeys):
#             k1, k2 = ckey
#             # stack[j] = np.fft.irfft(np.conj(fdat[k1]) * fdat[k2])
#             cc_stack_freq[j] += np.conj(fdat[k1]) * fdat[k2]

#         t0 += cclen_td
#         t1 += cclen_td
#         nstack += 1

#     nlag = int(keeplag * sr)
#     stack = np.zeros((ncc, nlag * 2), dtype=np.float32)

#     for i in range(ncc):
#         cc = np.real(np.fft.irfft(cc_stack_freq[i])) / nstack
#         stack[i, :nlag] = cc[-nlag:]
#         stack[i, nlag:] = cc[:nlag]

#     return stack, ckeys


# def linear_regression_old(xv, yv, weights, outlier_sd=None):
#     c, stats = polyfit(xv, yv, 1, full=True, w=weights)
#     yint, slope = c
#     residual = stats[0][0] / len(xv)
#     ikeep = np.arange(len(xv))

#     if outlier_sd is not None:
#         residuals = np.abs(yv - (yint + slope * xv))
#         std_res = np.std(residuals)
#         ikeep = np.where(residuals < std_res * outlier_sd)[0]
#         if len(ikeep) != 0:
#             c, stats = polyfit(xv[ikeep], yv[ikeep], 1, full=True, w=weights[ikeep])
#             yint, slope = c
#             residual = stats[0][0] / len(xv)
#             print("stdev residuals: %.3f" % residual)

#     return yint, slope, residual, ikeep


# def linear_regression2(xv, yv, weights, outlier_val=0.0025):
#     ikeep = np.where(np.abs(yv / xv) < outlier_val)[0]
#     c, stats = polyfit(xv[ikeep], yv[ikeep], 1, full=True, w=weights[ikeep])
#     yint, slope = c
#     residual = stats[0][0] / len(xv)

#     return yint, slope, residual, ikeep


# def linear_regression3(xv, yv, weights):
#     XX = np.vstack((xv, np.ones_like(xv))).T
#     # out = np.linalg.lstsq(XX[:, :-1], yv, rcond=None)
#     # out = np.linalg.lstsq(XX, yv, rcond=None)
#     out = np.linalg.lstsq(XX * weights[:, None], yv * weights, rcond=None)
#     print(out)
#     c = out[0]
#     residual = out[1] / len(xv)
#     slope, yint = c

#     return yint, slope, residual


# def linear_regression_yzero(xv, yv, weights):
#     XX = np.vstack((xv, np.ones_like(xv))).T
#     # out = np.linalg.lstsq(XX[:, :-1], yv, rcond=None)
#     out = np.linalg.lstsq(XX, yv, rcond=None)
#     # c, stats =
#     print(out)
#     c = out[0]
#     residual = out[1] / len(xv)

#     # # c, stats = polyfit(xv[ikeep], yv[ikeep], 1, full=True, w=weights[ikeep])
#     # yint, slope = c
#     slope, yint = c
#     # residual = stats[0][0]
#     return yint, slope, residual
