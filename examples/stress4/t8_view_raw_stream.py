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
from microquake.core import read

plt.ion()

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations = pickle.load(f)
###############################

sr_raw = 6000.0
dsr = 1000.0
# whiten_freqs = np.array([60, 80, 320, 350])
fband = [50, dsr / 2]
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
plt.plot(tr.data)

tr = stream[1]
tr.data = np.roll(dat, 100)
plt.plot(tr.data)


nchan = len(stream)
nsamp_raw = np.max([len(tr.data) for tr in stream])
nsamp = nsamp_raw // decf
data = np.zeros((nchan, nsamp), dtype=np.float32)

chan_names = []

for i, tr in enumerate(stream):
    print(tr)
    tr.data -= np.nanmean(tr.data)
    tr.data = xutil.linear_detrend_nan(tr.data)
    tr.data = np.nan_to_num(tr.data)
    tr.filter('bandpass', freqmin=fband[0], freqmax=fband[1])
    # sig = np.sign(tr.data[::decf])
    sig = tr.data[::decf]
    i0 = int((tr.stats.starttime - req_start_time) * dsr + 0.5)
    slen = min(len(sig), nsamp - i0)
    data[i, i0: i0 + slen] = sig[:slen]
    chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

chan_names = np.array(chan_names)
# data = np.sign(data).astype(np.bool_)

plt.plot(data[0])
plt.plot(data[1])

