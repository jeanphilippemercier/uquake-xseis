from importlib import reload
import os
import numpy as np
import os
import time
# import h5py
# from glob import glob
from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange_workflow as flow
from xseis2 import xchange
import matplotlib.pyplot as plt
import pickle
from microquake.core import read

plt.ion()

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations = pickle.load(f)
###############################

sr_raw = 6000.0
dsr = 1000.0
# whiten_freqs = np.array([60, 80, 320, 350])
fband = [50, dsr / 2]
whiten_freqs = np.array([40, 50, 380, 400])
decf = int(sr_raw / dsr)


fname = os.path.join(os.environ['SPP_COMMON'], "cont10min.mseed")
stream = read(fname)
stream.sort()
req_start_time = stream[0].stats.starttime

tr = stream[0]
dat = tr.data.copy()
tr.data += np.linspace(0, np.max(tr.data), len(tr.data))
tr.data[:10000] = np.nan
tr.data[50000:1000000] = np.nan

tr = stream[1]
tr.data = np.roll(dat, 100)
# plt.plot(tr.data)

# nan_ind = np.where(np.isnan(tr.data))[0]
# tr.data[nan_ind] = 0
# plt.plot(tr.data)
# plt.plot(tr.data)

nchan = len(stream)
nsamp_raw = np.max([len(tr.data) for tr in stream])
nsamp = nsamp_raw // decf
data = np.zeros((nchan, nsamp), dtype=np.float32)


chan_names = []

for i, tr in enumerate(stream):
    print(tr)
    sr = tr.stats.sampling_rate
    sig = tr.data
    ixnan = np.where(np.isnan(sig))[0]
    sig -= np.nanmean(sig)
    sig = xutil.linear_detrend_nan(sig)
    sig[ixnan] = 0
    # sig = xutil.bandpass(sig, fband, sr)
    sig = xutil.whiten_sig(sig, sr, whiten_freqs, pad_multiple=1000)
    sig[ixnan] = 0
    # sig = np.sign(tr.data[::decf])
    sig = sig[::decf]
    sig = np.sign(sig)
    sig[sig < 0] = 0

    i0 = int((tr.stats.starttime - req_start_time) * dsr + 0.5)
    slen = min(len(sig), nsamp - i0)
    data[i, i0: i0 + slen] = sig[:slen]
    chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

chan_names = np.array(chan_names)
data = data.astype(np.bool_)

req_start_time = tr.stats.starttime
tstring = req_start_time.__str__()
fname = os.path.join(os.environ['SPP_COMMON'], f"ob_data_{tstring}.npz")

np.savez_compressed(fname, start_time=tstring, data=data, sr=dsr, chans=chan_names)

data, sr, starttime, endtime, chan_names = flow.load_npz_continuous(fname)


plt.plot(data[1])
plt.plot(data[0])


# sig = data[0]
xplot.freq(data[0], dsr)
xplot.freq(data[1], dsr)


imax, vmax, cc = xchange.measure_shift_cc(data[0], data[1])
print(imax, vmax)
plt.plot(cc)










########################

reload(xutil)
tr = stream[0]
sr = tr.stats.sampling_rate
# sig = np.zeros(xutil.nextpow2(len(tr.data)), dtype=np.float32)
sig = tr.data.copy()
sig -= np.nanmean(sig)
sig = xutil.linear_detrend_nan(sig)
sig = np.nan_to_num(sig)
# sig = xutil.bandpass(sig, fband, sr)
sig2 = xutil.whiten_sig(sig, sr, whiten_freqs, pad_multiple=1000)
# pad = xutil.nextpow2(len(sig))
# pad = roundup(len(sig), nearest_multiple=1000)
# whiten_win = xutil.freq_window(whiten_freqs, pad, sr)
plt.plot(xutil.maxnorm(sig))
plt.plot(xutil.maxnorm(sig2))

plt.plot(sig2)

xplot.freq(sig, sr)
xplot.freq(sig2, sr)





#######################
sig = data[0]
xplot.freq(sig, dsr)

import math


def roundup(x, nearest_multiple):
    return int(math.ceil(x / nearest_multiple)) * int(nearest_multiple)


# roundup(len(sig), nearest_multiple=100)
roundup(101, nearest_multiple=1000)

tr = stream[0]
sr = tr.stats.sampling_rate
# sig = np.zeros(xutil.nextpow2(len(tr.data)), dtype=np.float32)
sig = tr.data.copy()
sig -= np.nanmean(sig)
sig = xutil.linear_detrend_nan(sig)
sig = np.nan_to_num(sig)
# sig = xutil.bandpass(sig, fband, sr)
# sig = xutil.whiten_sig(sig, sr, whiten_freqs)

# pad = xutil.nextpow2(len(sig))
pad = roundup(len(sig), nearest_multiple=1000)
whiten_win = xutil.freq_window(whiten_freqs, pad, sr)
fsig = np.fft.rfft(sig, n=pad)

fsig = whiten_win * xutil.phase(fsig)
sig2 = np.fft.irfft(fsig, n=pad)[:len(sig)]

plt.plot(xutil.maxnorm(sig))
plt.plot(xutil.maxnorm(sig2))
plt.plot(sig2)

xplot.freq(sig, sr)
xplot.freq(sig2, sr)

