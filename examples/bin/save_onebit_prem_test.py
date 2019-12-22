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

sites = [18, 20, 22, 24, 26, 28, 30, 32, 41, 44, 46, 48, 50, 52, 54, 56, 58, 63, 65, 67, 69, 71, 79, 81, 90, 104, 106, 108, 110, 112, 115, 117, 119, 121, 126]

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

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
req_start_time = UTCDateTime(curtime + req_lag)
req_end_time = UTCDateTime(req_start_time + req_length)

tstring = req_start_time.__str__()
# fname = os.path.join(os.environ['SPP_COMMON'], "data_dump", f"ob_data_{tstring}")
fname = os.path.join(os.environ['SPP_COMMON'], "phil_misc", f"ob_data_{tstring}")
data = np.ones(500)
np.savez_compressed(fname, data=data)
