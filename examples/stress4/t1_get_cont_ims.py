import os
import numpy as np
import h5py
from importlib import reload
from datetime import datetime, timedelta

from xseis2 import xutil
from glob import glob
from pytz import utc
# starttime

import time
from microquake.core.settings import settings
from microquake.core.stream import Trace, Stream

# from xseis2.h5stream import H5Stream

from microquake.clients.ims import web_client
# from microquake.clients.api_client import (post_data_from_objects,
                                           # get_event_by_id)
from obspy import UTCDateTime
# from microquake.core.helpers.time import get_time_zone
# import json

sites = [int(station.code) for station in settings.inventory.stations()]
base_url = settings.get('ims_base_url')
api_base_url = settings.get('api_base_url')
inventory = settings.inventory
network_code = settings.NETWORK_CODE

t0 = time.time()

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
endtime = curtime + timedelta(minutes=-120)
starttime = endtime - timedelta(minutes=1)
# st = get_continuous_data(starttime, endtime)
st = web_client.get_continuous(base_url, starttime, endtime,
                               sites, utc, network=network_code)


elapsed = time.time() - t0
print('elapsed: %.2f sec' % elapsed)

[tr.stats.station for tr in st]
[len(tr.data) / 6000 for tr in st]


######################################

t0 = time.time()

curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
endtime = curtime + timedelta(minutes=-120)
starttime = endtime - timedelta(minutes=10)
# st = get_continuous_data(starttime, endtime)
st = web_client.get_continuous(base_url, starttime, endtime, sites[100], utc, network=network_code)

elapsed = time.time() - t0
print('elapsed: %.2f sec' % elapsed)
