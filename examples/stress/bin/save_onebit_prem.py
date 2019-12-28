import os
import numpy as np
import h5py
from importlib import reload
from datetime import datetime, timedelta

from xseis2 import xutil
from glob import glob
from pytz import utc

import time
from microquake.core.settings import settings
from microquake.core.stream import Trace, Stream
# from microquake.clients.ims import web_client
from xseis2 import web_client

from obspy import UTCDateTime
# from microquake.core.helpers.time import get_time_zone


stations = settings.inventory.stations()
# sites = [int(sta.code) for sta in stations]
base_url = settings.get('ims_base_url')
api_base_url = settings.get('api_base_url')
inventory = settings.inventory
network_code = settings.NETWORK_CODE

sites = [18, 20, 22, 24, 26, 28, 30, 32, 41, 44, 46, 48, 50, 52, 54, 56, 58, 63, 65, 67, 69, 71, 79, 81, 90, 104, 106, 108, 110, 112, 115, 116, 117, 119, 121, 126, 140]

# import pickle
# fname = os.path.join(os.environ['SPP_COMMON'], "stations.pickle")
# with open(fname, 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(stations, output, pickle.HIGHEST_PROTOCOL)


#############################

sr_raw = 6000.0
dsr = 2000.0
whiten_freqs = np.array([40, 50, 390, 400])
# fband = [50, 400]
req_lag = timedelta(hours=-3)
req_length = timedelta(minutes=10)
decf = int(sr_raw / dsr)

# while True:

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
req_start_time = UTCDateTime(curtime + req_lag)
req_end_time = UTCDateTime(req_start_time + req_length)

stream = web_client.get_continuous(base_url, req_start_time, req_end_time, sites, utc, network=network_code, format='binary')

# if stream is None:
#     continue
# sta_ids = np.array([sta.code for sta in stations])[::5]
# stream = flow.get_continuous_fake(req_start_time, req_end_time, sta_ids, sr=sr_raw)
stream.sort()

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

    # izero = np.where(sig == 0)[0]
    # sig[izero] = np.random.choice([-1, 1], len(izero))

chan_names = np.array(chan_names)
data = data.astype(np.bool_)

tstring = req_start_time.__str__()
fname = os.path.join(os.environ['SPP_COMMON'], "data_dump", f"ob_data_{tstring}")

np.savez_compressed(fname, start_time=tstring, data=data, sr=dsr, chans=chan_names)
