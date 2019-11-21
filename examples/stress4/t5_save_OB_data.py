from importlib import reload
import os
import numpy as np
import os
import time
# import h5py
# from glob import glob
from datetime import datetime, timedelta

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import VelChange, XCorr, ChanPair, Base

from loguru import logger
from obspy import UTCDateTime
# from microquake.core.settings import settings
# from microquake.core.stream import Trace, Stream
# from pytz import utc
from microquake.plugin.site.core import read_csv
import matplotlib.pyplot as plt
from pytz import utc


plt.ion()

filename = "/home/phil/data/oyu/spp_common/sensors.csv"
inv = read_csv(filename, site_code='', has_header=True)
stations = inv.stations()
stations = sorted(stations, key=lambda x: x.code)

sr_raw = 6000.0
dsr = 1000.0
# whiten_freqs = np.array([60, 80, 320, 350])
fband = [50, dsr / 2]
req_lag = timedelta(hours=-3)
req_length = timedelta(minutes=10)
decf = int(sr_raw / dsr)

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
req_start_time = UTCDateTime(curtime + req_lag)
req_end_time = UTCDateTime(req_start_time + req_length)

sta_ids = np.array([sta.code for sta in stations])[::5]
stream = flow.get_continuous_fake(req_start_time, req_end_time, sta_ids, sr=sr_raw)
stream.sort()

nchan = len(stream)
nsamp_raw = np.max([len(tr.data) for tr in stream])
nsamp = nsamp_raw // decf
data = np.zeros((nchan, nsamp), dtype=np.float32)

chan_names = []

for i, tr in enumerate(stream):
    print(tr)
    # sr = tr.stats.sampling_rate
    tr.filter('bandpass', freqmin=fband[0], freqmax=fband[1])
    sig = np.sign(tr.data[::decf])
    # sig = tr.data[::decf]
    i0 = int((tr.stats.starttime - req_start_time) * dsr + 0.5)
    slen = min(len(sig), nsamp - i0)
    data[i, i0: i0 + slen] = sig[:slen]
    chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

chan_names = np.array(chan_names)
data = np.sign(data).astype(np.bool_)

tstring = req_start_time.__str__()
fname = os.path.join(os.environ['SPP_COMMON'], f"ob_data_{tstring}")

np.savez_compressed(fname, start_time=tstring, data=data, sr=dsr, chans=chan_names)
