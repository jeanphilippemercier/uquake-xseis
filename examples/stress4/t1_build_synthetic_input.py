import os
from datetime import datetime, timedelta
import numpy as np
import os
import h5py
from importlib import reload

from xseis2 import xutil
from glob import glob

from microquake.core.settings import settings

from obspy import UTCDateTime

from microquake.core.stream import Trace, Stream
from pytz import utc


def get_continuous_fake(start_time, end_time, stations, fband=[20, 400], sr=1000.0):
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

    nsamp = int((end_time - start_time) * sr)
    traces = []
    for sta in stations:
        # loc = sta.loc
        dat = xutil.band_noise(fband, sr, nsamp)
        for chan in sta.channels:
            # chans.append(f"{sta.code}.{chan.code}")
            # locs.append(loc)
            tr = Trace(data=dat)
            tr.stats.starttime = start_time
            tr.stats.sampling_rate = sr
            tr.stats.channel = chan.code
            tr.stats.station = sta.code
            traces.append(tr)

    st = Stream(traces=traces)
    return st


stations = settings.inventory.stations()

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)

# endtime = curtime + timedelta(minutes=-120)
# starttime = endtime - timedelta(minutes=10)
stacklen = 600
times = [curtime + timedelta(seconds=i * stacklen) for i in range(5)]

start_time = times[0]
end_time = start_time + timedelta(seconds=stacklen)

# for t0 in times:
# print(t0)
# t1 = t0 + timedelta(seconds=stacklen)

stream = get_continuous_fake(start_time, end_time, stations)
stream.filter('bandpass', freqmin=100, freqmax=1000)
data, sr, t0 = stream.as_array(stacklen)


# import redis
# 10hr_1000hz

stations = settings.inventory.stations()
station = stations[0]
st_id = station.code
st_loc = station.loc

chans = []
locs = []
for sta in stations:
    loc = sta.loc
    for chan in sta.channels:
        chans.append(f"{sta.code}.{chan.code}")
        locs.append(loc)
chans = np.array(chans)
locs = np.array(locs)


nchan = 380
# nchan = len(chans)
sr = 1000
hours = 0.2
nsamp = nchan * sr * hours * 3600
size = xutil.sizeof(nsamp * 4.0)
print(size)


sites = [station.code for station in stations]
site_locs = [station.loc for station in stations]
ldict = dict(zip(sites, site_locs))


dists = dict()
for ck in ckeys:
    c1, c2 = ck.split('_')
    dd = xutil.dist(ldict[c1[:-2]], ldict[c2[:-2]])
    dists[ck] = dd

rows = []
for k, v in dists.items():
    rows.append(ChanPair(ckey=k, dist=v))
    # print(k, v)



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
