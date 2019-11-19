import os
from datetime import datetime, timedelta
import numpy as np
import os
import h5py
from importlib import reload

from xseis2 import xutil
from xseis2 import xchange

from glob import glob

from microquake.core.settings import settings

from obspy import UTCDateTime

from microquake.core.stream import Trace, Stream
from pytz import utc
import time
from microquake.core.util import tools

from xseis2.xutil import build_slice_inds


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

# chans = hstream.channels
# sr_raw = hstream.samplerate

dsr = 1000.0
# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 20.0
stepsize = cclen
# stepsize = cclen / 2
keeplag = 1.0
stacklen = 600.0
onebit = True

t0 = time.time()

# ###############
# # stream = get_continuous_fake(start_time, end_time, stations, sr=6000.0)
# stream = get_continuous_fake(start_time, end_time, stations[:10], sr=6000.0)
# for tr in stream:
#     print(tr)
#     sr = tr.stats.sampling_rate
#     if sr > dsr:
#         tr.filter('lowpass', freq=dsr / 2)
#         tr.decimate(int(sr / dsr))
# ####################

stream = get_continuous_fake(start_time, end_time, stations[::2], sr=dsr)
data, sr, time0 = stream.as_array(stacklen)
chans = np.array([f"{tr.stats.station}.{tr.stats.channel}" for tr in stream])

ckeys = xutil.unique_pairs(np.arange(len(chans)))
ckeys = xutil.ckeys_remove_intersta(ckeys, chans)
print(len(ckeys))

dc = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys, cclen, keeplag, stepsize=stepsize, whiten_freqs=whiten_freqs, onebit=onebit)


print('elapsed: %.2f sec' % time.time() - t0)

# wlen = stacklen
# nrow, ncol = data.shape
# slices = build_slice_inds(0, ncol, wlen, stepsize=stepsize)












###############################





def array_slice_gen(dat, wlen, stepsize):
    nrow, ncol = dat.shape
    slices = build_slice_inds(0, ncol, wlen, stepsize=stepsize)
    for i, sl in enumerate(slices):
        yield dat[:, sl[0]:sl[1]]


datgen = array_slice_gen(data, cclen, stepsize)

a = next(datgen)

dc, ckeys_ix = xchange.xcorr_stack_slices(
    datgen, chans, cclen, sr_raw, sr, keeplag, whiten_freqs, onebit=onebit)
ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

pipe = rhandle.pipeline()
rows = []
for i, sig in enumerate(dc):
    # print(i)
    ckey = ckeys[i]
    dkey = f"{str(t0)} {ckey}"
    pipe.set(dkey, xio.array_to_bytes(sig))
    rows.append(XCorr(time=t0, ckey=ckey, data=dkey))

pipe.execute()  # add data to redis
session.add_all(rows)  # add rows to sql
session.commit()












############################
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
