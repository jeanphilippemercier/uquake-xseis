import statsmodels.api as sm
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.fftpack
import math

from scipy import interpolate


def cosine_taper(npts, p=0.1):

    frac = int(npts * p / 2.0 + 0.5)
    idx1 = 0
    idx2 = frac - 1
    idx3 = npts - frac
    idx4 = npts - 1

    cos_win = np.zeros(npts)
    cos_win[idx1:idx2 + 1] = 0.5 * (
        1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) /
                      (idx2 - idx1))))
    cos_win[idx2 + 1:idx3] = 1.0
    cos_win[idx3:idx4 + 1] = 0.5 * (
        1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) /
                      (idx4 - idx3))))

    # if indices are identical division by zero
    # causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0

    return cos_win


def stretch(sig, sr, tt_change_percent, len_zpad=1000):

    npts = len(sig)
    tmin, tmax = np.array([-npts, npts]) / 2. / sr
    dt_new = 1. / (sr - tt_change_percent / 100. * sr)

    hlen = npts / 2 + len_zpad
    x = np.arange(-hlen, hlen) / sr

    psig = np.zeros(npts + len_zpad * 2)
    psig[len_zpad: -len_zpad] = sig

    xnew = np.arange(-npts / 2, npts / 2) * dt_new

    interp = interpolate.interp1d(x, psig)
    newsig = interp(xnew)

    return newsig


def nextpow2_power(val):
    return np.ceil(np.log2(np.abs(val)))


def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
    window_len = 2*half_win+1
    # extending the data at beginning and at the end
    # to apply the window at the borders
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.hanning(window_len).astype('complex')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[half_win:len(y)-half_win]


def getCoherence(dcs, ds1, ds2):
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    coh[valids] = dcs[valids]/(ds1[valids]*ds2[valids])
    coh[coh > (1.0+0j)] = 1.0+0j
    return coh


def mwcs(ccCurrent, ccReference, fmin, fmax, sampRate, trange, windL, step,
         plot=False):
    """...

    :type ccCurrent: :class:`numpy.ndarray`
    :param ccCurrent: The "Current" timeseries
    :type ccReference: :class:`numpy.ndarray`
    :param ccReference: The "Reference" timeseries
    :type fmin: float
    :param fmin: The lower frequency bound to compute the dephasing
    :type fmax: float
    :param fmax: The higher frequency bound to compute the dephasing
    :type sampRate: float
    :param sampRate: The sample rate of the input timeseries
    :type tmin: float
    :param tmin: The leftmost time lag (used to compute the "time lags array")
    :type windL: float
    :param windL: The moving window length
    :type step: float
    :param step: The step to jump for the moving window
    :type plot: bool
    :param plot: If True, plots the MWCS result for each window. Defaults to
        False

    :rtype: :class:`numpy.ndarray`
    :returns: [Taxis,deltaT,deltaErr,deltaMcoh]. Taxis contains the central
        times of the windows. The three other columns contain dt, error and
        mean coherence for each window.
    """

    # windL = np.int(windL*sampRate)
    # step = np.int(step*sampRate)
    count = 0
    deltaT = []
    deltaErr = []
    deltaMcoh = []
    Taxis = []
    padd = 2**(nextpow2_power(windL)+2)

    # Tentative checking if enough point are used to compute the FFT
    freqVec = scipy.fftpack.fftfreq(int(padd), 1./sampRate)[:int(padd)//2]
    indRange = np.argwhere(np.logical_and(freqVec >= fmin,
                                          freqVec <= fmax))
    if len(indRange) < 2:
        padd = 2**(nextpow2_power(windL)+3)

    tp = cosine_taper(windL, .85)

    # timeaxis = (np.arange(len(ccCurrent)) / float(sampRate))+tmin
    npts = len(ccCurrent)
    timeaxis = np.arange(-npts / 2, npts / 2) / sampRate
    tmin, tmax = trange

    minind = int(tmin * sampRate + npts / 2) - (windL / 2)
    imax_stop = int(tmax * sampRate + npts / 2) + (windL / 2)
    maxind = windL

    # print minind, imax_stop
    # while maxind <= len(ccCurrent):
    # while maxind <= imax_stop:
    while minind + windL <= imax_stop:
        ind = minind
        # print maxind, ind, (ind+windL)
        cci = ccCurrent[ind:(ind + windL)].copy()
        cci = scipy.signal.detrend(cci, type='linear')
        cci -= cci.min()
        cci /= cci.max()
        cci -= np.mean(cci)
        cci *= tp

        cri = ccReference[ind:(ind+windL)].copy()
        cri = scipy.signal.detrend(cri, type='linear')
        cri -= cri.min()
        cri /= cri.max()
        cri -= np.mean(cri)
        cri *= tp

        Fcur = scipy.fftpack.fft(cci, n=int(padd))[:int(padd)//2]
        Fref = scipy.fftpack.fft(cri, n=int(padd))[:int(padd)//2]

        Fcur2 = np.real(Fcur)**2 + np.imag(Fcur)**2
        Fref2 = np.real(Fref)**2 + np.imag(Fref)**2

        smoother = 5

        dcur = np.sqrt(smooth(Fcur2, window='hanning', half_win=smoother))
        dref = np.sqrt(smooth(Fref2, window='hanning', half_win=smoother))

        # Calculate the cross-spectrum
        X = Fref*(Fcur.conj())
        X = smooth(X, window='hanning', half_win=smoother)
        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freqVec = scipy.fftpack.fftfreq(len(X)*2, 1./sampRate)[:int(padd)//2]
        indRange = np.argwhere(np.logical_and(freqVec >= fmin,
                                              freqVec <= fmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[indRange])

        # Get Weights
        w = 1.0 / (1.0 / (coh[indRange]**2) - 1.0)
        w[coh[indRange] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[indRange]))
        # w /= (np.sum(w)/len(w)) #normalize
        w = np.real(w)

        # Frequency array:
        v = np.real(freqVec[indRange]) * 2 * np.pi
        vo = np.real(freqVec) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[indRange]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin
        # weights for the WLS must be the variance !
        res = sm.regression.linear_model.WLS(phi, v, w**2).fit()

        # print "forced", np.real(res.params[0])
        # print "!forced", np.real(res2.params[0])

        m = np.real(res.params[0])
        deltaT.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi-m*v)**2) / (np.size(v)-1)
        s2x2 = np.sum(v**2 * w**2)
        sx2 = np.sum(w * v**2)
        e = np.sqrt(e*s2x2 / sx2**2)

        deltaErr.append(e)
        deltaMcoh.append(np.real(mcoh))
        Taxis.append(timeaxis[ind + windL // 2])
        count += 1

        minind += step
        maxind += step
        del Fcur, Fref
        del X
        del freqVec
        del indRange
        del w, v, e, s2x2
        del res

    if maxind > len(ccCurrent) + step:
        logging.warning("The last window was too small, but was computed")

    return np.array([Taxis, deltaT, deltaErr, deltaMcoh]).T
