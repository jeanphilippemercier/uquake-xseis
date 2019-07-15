import matplotlib.pyplot as plt

import numpy as np
# import datetime
# import struct
from obspy.core import UTCDateTime
import os
import glob
# import pickle
from xseis2 import xutil
# from xseis import xio
import h5py
import time

# from obspy import read
# from obspy.io.rg16.core import _read_rg16
from microquake.core import read
from spp.core.settings import settings
from glob import glob
from importlib import reload

plt.ion()


ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
outdir = ddir
hfname = os.path.join(outdir, '10hr_sim.h5')


fles = np.sort(glob(os.path.join(ddir, 'dec_*.mseed')))

#########################
stream = read(fles[0])
sr = stream[0].stats.sampling_rate
names = np.array([f"{tr.stats.station}.{tr.stats.channel}" for tr in stream])
names = np.sort(names)
ndict = dict(zip(names, np.arange(len(names))))

lens = np.array([len(tr) for tr in stream])
nsamp = np.max(lens) * len(fles)
nchan = len(stream)

t0 = stream[0].stats.starttime

# hf_out.close()


# from datetime import timedelta, datetime
# import os
# # from xseis2 import xutil
# import h5py

from xseis import xplot
from xseis import xutil
from xseis import xio
# from xseis2 import xutil
# from importlib import reload


# exit
# ipython
plt.ion()


sites = [station.code for station in settings.inventory.stations()]
site_locs = [station.loc for station in settings.inventory.stations()]
ldict = dict(zip(sites, site_locs))

locs = []
for sta in names:
    locs.append(ldict[sta.split('.')[0]])

locs = np.array(locs, dtype=np.float32)

xplot.stations(locs)


# seconds = 3600.
# src_times = np.array([5, 5])
# src_durs = np.array([3000, 3000])

# freqs = [2, 5, 20, 23]
freqs = [20, 40, 490, 500]
vel = 3200.
# seconds = 60
sr = 1000.

# src_locs = np.array([[3000, 1500, 400]], dtype=np.float32)
src_locs = np.array([locs[0]], dtype=np.float32)
src_times = np.array([0.5])
src_durs = np.array([3600 * 3])
seconds = 3600 * 5

# src_locs = np.array([[3000, 1500, 400], [3000, 2000, 150]], dtype=np.float32)
# src_times = np.array([5, 20])
# src_durs = np.array([5, 8])
# seconds = 30

nchan = len(locs)
nsamp = int(seconds * sr)

reload(xutil)
size = xutil.sizeof(nchan * nsamp * 4.0)
print("%.2f %s" % (size[0], size[1]))


# dd = xutil.dist2many(src_locs[0], locs)
# tts = (dd / vel * sr).astype(int)
# imax = np.max(tts)
# print("tt imax:", imax)

print("nchan:", nchan)
print("len(noise)", nsamp)
# print("len(sig)", nsamp_sig)
noise_scale = 10
taplen = 0.001

i = 0
nsamp_sig = int(src_durs[i] * sr)
noise = xutil.noise1d(nsamp, freqs, sr, scale=noise_scale, taplen=taplen)
sig = xutil.noise1d(nsamp_sig, freqs, sr, scale=1, taplen=taplen)

plt.plot(sig[:100000])
plt.plot(noise[:100000])


hf_out = h5py.File(hfname, 'w')
dset = hf_out.create_dataset("data", (nchan, nsamp), dtype=np.float32)
# print(list(hf.keys()))
# hf_out.create_dataset('sta_locs', data=lkeep.astype(np.float32))
hf_out.create_dataset('channels', data=names.astype('S15'))
# hf_out.create_dataset('chan_map', data=np.arange(nchan, dtype=np.uint16))
hf_out.attrs['samplerate'] = float(sr)
hf_out.attrs['starttime'] = str(t0)
hf_out.create_dataset('locs', data=locs.astype(np.float32))


# dat = np.zeros((nchan, nsamp), dtype=np.float32)
for i in range(nchan):
    dset[i] = xutil.noise1d(nsamp, freqs, sr, scale=noise_scale, taplen=taplen)

for i, sloc in enumerate(src_locs):
    print(i)

    nsamp_sig = int(src_durs[i] * sr)
    sig = xutil.noise1d(nsamp_sig, freqs, sr, scale=1, taplen=taplen)

    dd = xutil.dist2many(sloc, locs)
    tts = (dd / vel * sr + 0.5).astype(int)
    t0 = int(src_times[i] * sr)
    for j, tt in enumerate(tts):
        i0 = tt + t0
        dset[j, i0: i0 + nsamp_sig] = dset[j, i0: i0 + nsamp_sig][:] + sig

hf_out.close()


