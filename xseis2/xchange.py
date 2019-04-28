import numpy as np
from scipy import interpolate
import logging

import scipy.fftpack
import scipy.optimize
import scipy.signal
from scipy.stats import scoreatpercentile
from obspy.signal.invsim import cosine_taper
from scipy.fftpack.helper import next_fast_len
from obspy.signal.regression import linear_regression as obspy_linear_regression
from xseis2 import xutil
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt


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


def linear_regression(xv, yv, weights, outlier_sd=None):
    c, stats = polyfit(xv, yv, 1, full=True, w=weights)
    yint, slope = c
    residual = stats[0][0] / len(xv)
    ikeep = np.arange(len(xv))

    if outlier_sd is not None:
        residuals = np.abs(yv - (yint + slope * xv))
        std_res = np.std(residuals)
        ikeep = np.where(residuals < std_res * outlier_sd)[0]
        if len(ikeep) != 0:
            c, stats = polyfit(xv[ikeep], yv[ikeep], 1, full=True, w=weights[ikeep])
            yint, slope = c
            residual = stats[0][0] / len(xv)
            print("stdev residuals: %.3f" % residual)

    return yint, slope, residual, ikeep


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

    return fdat, filt


def measure_shift_fwins_cc(fwins1, fwins2, interp_factor=1):

    nwin, nfreq = fwins1.shape
    pad = nfreq * 2 - 1

    out = np.zeros((nwin, 2))

    for i, (w1, w2) in enumerate(zip(fwins1, fwins2)):
        ccf = np.conj(w1) * w2
        cc = np.fft.irfft(ccf, n=pad * interp_factor)
        cc = np.roll(cc, len(cc) // 2)
        imax = (np.argmax(cc) - len(cc) // 2) / interp_factor
        out[i] = [imax, np.max(cc) * interp_factor]

    return out


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
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
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

    return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T


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


