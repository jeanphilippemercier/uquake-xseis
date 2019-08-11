from datetime import timedelta

import numpy as np

import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload

from xseis2 import xutil
from xseis2 import xplot
# from xseis2.xchange import smooth, nextpow2, getCoherence
from obspy.core import UTCDateTime

from microquake.io.h5stream import H5Stream


plt.ion()

ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
hfname = os.path.join(ddir, '10hr_1000hz.h5')


hs = H5Stream(hfname)
hs.starttime
hs.endtime
chans = hs.channels

# hs.get_row_indexes(chans[1:10])

t0 = hs.starttime
t1 = t0 + timedelta(hours=1)

dat = hs.query(chans, t0, t1)



# dat = hs.dset[[1, 0, 4, 10]]

hf = h5py.File(hfname, 'r')
names = hf['channels'][:].astype(str)
locs = hf['locs'][:]
sr = hf.attrs['samplerate']
dset = hf['data']
tmin = UTCDateTime(hf.attrs['starttime'])
tmax = tmin + timedelta(seconds=dset.shape[1] / sr)
reload(xutil)

from xseis2.xutil import bandpass, freq_window, phase

# PARAMS - config
whiten_freqs = np.array([80, 100, 260, 300])
cclen = int(10 * sr)
keeplag = int(1.5 * sr)
stepsize = cclen
onebit = True

# Params to function
t0 = tmin
t1 = tmin + timedelta(hours=1)

channels = names

# def compute_cc_stack(t0, t1, channels=None):

reload(xutil)
ckeys = xutil.unique_pairs(np.arange(len(names)))
ckeys = xutil.ckeys_remove_intersta(ckeys, names)

i0 = int((t0 - tmin) * sr)
i1 = int((t1 - tmin) * sr)
slices = xutil.build_slice_inds(i0, i1, cclen, stepsize=stepsize)


def xcorr_stack_slices(dset, sr, ckeys, cclen, keeplag, whiten_freqs, onebit=True):

    ncc = len(ckeys)
    padlen = cclen * 2
    nfreq = int(padlen // 2 + 1)

    whiten_win = None
    if whiten_freqs is not None:
        whiten_win = freq_window(whiten_freqs, padlen, sr)

    fstack = np.zeros((ncc, nfreq), dtype=np.complex64)

    for i, sl in enumerate(slices):
        print(f"{i} / {len(slices)}")
        dat = dset[:, sl[0]:sl[1]]

        if onebit is True:
            dat = bandpass(dat, whiten_freqs[[0, -1]], sr)
            dat = np.sign(dat)

        fdat = np.fft.rfft(dat, axis=1, n=padlen)

        for irow in range(fdat.shape[0]):
            fdat[irow] = whiten_win * phase(fdat[irow])

        for j, ckey in enumerate(ckeys):
            k1, k2 = ckey
            # stack[j] = np.fft.irfft(np.conj(fdat[k1]) * fdat[k2])
            fstack[j] += np.conj(fdat[k1]) * fdat[k2]

    stack = np.zeros((ncc, keeplag * 2), dtype=np.float32)

    for i in range(ncc):
        cc = np.real(np.fft.irfft(fstack[i])) / len(slices)
        stack[i, :keeplag] = cc[-keeplag:]
        stack[i, keeplag:] = cc[:keeplag]


dd = xutil.dist_diff_ckeys(ckeys, locs)
isort = np.argsort(dd)
xplot.im(stack[isort])
