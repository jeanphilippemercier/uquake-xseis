import matplotlib.pyplot as plt

import numpy as np
# import datetime
# import struct
import os
# import pickle
from xseis2 import xutil
# from xseis import xio
import h5py

# from obspy import read
# from obspy.io.rg16.core import _read_rg16
from microquake.core import read
from microquake.core.settings import settings
from glob import glob
from importlib import reload

plt.ion()


ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
outdir = ddir
hfname = os.path.join(outdir, '10hr_1000hz.h5')

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

hf_out = h5py.File(hfname, 'w')
dset = hf_out.create_dataset("data", (nchan, nsamp), dtype=np.float32)
# print(list(hf.keys()))
# hf_out.create_dataset('sta_locs', data=lkeep.astype(np.float32))
hf_out.create_dataset('channels', data=names.astype('S15'))
# hf_out.create_dataset('chan_map', data=np.arange(nchan, dtype=np.uint16))
hf_out.attrs['samplerate'] = float(sr)

# fmt = '%Y/%m/%d %H:%M:%S'
# hf_out.attrs['time_fmt'] = fmt
hf_out.attrs['starttime'] = str(t0)

# t1 = UTCDateTime(str(t0))


reload(xutil)
for i, fle in enumerate(fles):
    print(i)
    stream = read(fle)

    for tr in stream:
        key = f"{tr.stats.station}.{tr.stats.channel}"
        irow = ndict[key]
        icol = int((tr.stats.starttime - t0) * sr)
        tlen = len(tr.data)
        xutil.nans_interp(tr.data)
        tr.detrend('linear')
        # tr.filter('bandpass', freqmin=20, freqmax=sr // 2)
        # tr.data -= np.mean(tr.data)
        dset[irow, icol: icol + tlen] = tr.data

# hf_out.close()

# sig1 = dset[1]
 # plt.plot(sig1)

sites = [station.code for station in settings.inventory.stations()]
site_locs = [station.loc for station in settings.inventory.stations()]
ldict = dict(zip(sites, site_locs))

locs = []
for sta in names:
    locs.append(ldict[sta.split('.')[0]])

locs = np.array(locs, dtype=np.float32)

hf_out.create_dataset('locs', data=locs.astype(np.float32))

hf_out.close()
