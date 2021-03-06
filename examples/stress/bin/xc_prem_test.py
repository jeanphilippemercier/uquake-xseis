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
from loguru import logger

from obspy import UTCDateTime
import redis
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


# ################################
# logger.info('Connect to psql database')
# db_string = "postgres://postgres:postgres@localhost"
# db = create_engine(db_string)
# print(db.table_names())
# session = sessionmaker(bind=db)()
# if overwrite:
#     flow.sql_drop_tables(db)
#     session.commit()
# Base.metadata.create_all(db)

##################################

# redis_pass = os.environ['SPP_REDIS_VELOCITY_PASSWORD']
# redis_url = os.environ['SPP_REDIS_VELOCITY_URL']
logger.info('Connect to redis database')
# rhandle = redis.Redis()
redis_url = 'redis://:8Arbqij73H@spp-redis-velocity-master:6379/0'
rhandle = redis.from_url(redis_url)
rhandle.keys()


rhandle = redis.Redis(host='localhost', port=6379, db=0)

if overwrite:
    rhandle.flushall()



#############################

sr_raw = 6000.0
dsr = 2000.0
whiten_freqs = np.array([40, 50, 390, 400])
# fband = [50, 400]
req_lag = timedelta(hours=-3)
req_length = timedelta(minutes=10)
decf = int(sr_raw / dsr)

while True:

    curtime = datetime.utcnow()
    curtime.replace(tzinfo=utc)
    req_start_time = UTCDateTime(curtime + req_lag)
    req_end_time = UTCDateTime(req_start_time + req_length)

    stream = web_client.get_continuous(base_url, req_start_time, req_end_time, sites, utc, network=network_code, format='binary')

    if stream is None:
        continue
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

###################################################################


# from microquake.core import read

# fname = os.path.join(os.environ['SPP_COMMON'], "cont10min.mseed")
# stream = read(fname)
# st2 = stream[::8]

# fname = os.path.join(os.environ['SPP_COMMON'], "cont10min_small.mseed")
# st2.write(fname)


# from microquake.core.helpers.timescale_db import get_continuous_data
# import time

# sr_raw = 6000.0
# dsr = 2000.0
# whiten_freqs = np.array([40, 50, 380, 400])
# # fband = [50, 400]
# req_lag = timedelta(hours=-3)
# # req_length = timedelta(minutes=10)
# req_length = timedelta(minutes=1)
# decf = int(sr_raw / dsr)


# curtime = datetime.utcnow()
# curtime.replace(tzinfo=utc)
# req_start_time = UTCDateTime(curtime + req_lag)
# req_end_time = UTCDateTime(req_start_time + req_length)

# t0 = time.time()

# st = get_continuous_data(req_start_time, req_end_time)

# elapsed = time.time() - t0
# print('elapsed: %.2f sec' % elapsed)

# [tr.stats.station for tr in st]
# [len(tr.data) / 6000 for tr in st]

# sensor_id = '1'
# st = get_continuous_data(starttime, endtime, sensor_id)


# curtime = datetime.utcnow()
# curtime.replace(tzinfo=utc)
# endtime = curtime + timedelta(minutes=-60)
# starttime = endtime - timedelta(minutes=5)
# print(starttime, endtime)
# st = get_continuous_data(starttime, endtime)
