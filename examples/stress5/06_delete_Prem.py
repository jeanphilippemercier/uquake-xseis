import os
import numpy as np
import h5py
from importlib import reload
from datetime import datetime, timedelta

from xseis2 import xutil
from glob import glob
from pytz import utc

# import time
# from microquake.core.settings import settings
# from microquake.core.stream import Trace, Stream
# from microquake.clients.ims import web_client

from obspy import UTCDateTime
from glob import glob

import os
import shutil

# from microquake.core.helpers.time import get_time_zone

flast = "ob_data_2019-12-03T14:51:54.099960Z.npz"

# ddir = os.path.join(os.environ['SPP_COMMON'], "data_dump")

data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
data_archive = os.path.join(os.environ['SPP_COMMON'], "data_archive")
# data_src = params['data_connector']['data_source']['location']
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))

found = False

for i, fname in enumerate(data_fles[:]):
    # print(f"processing {i} / {len(data_fles)}")
    basename = os.path.basename(fname)
    print(basename)
    if basename == flast:
        found = True
        print("found")

    shutil.move(fname, os.path.join(data_archive, basename))

    if found:
        break



# np.savez_compressed(fname, start_time=tstring, data=data, sr=dsr, chans=chan_names)

###################################################################


# from microquake.core import read

# fname = os.path.join(os.environ['SPP_COMMON'], "cont10min.mseed")
# stream = read(fname)
# st2 = stream[::8]

# fname = os.path.join(os.environ['SPP_COMMON'], "cont10min_small.mseed")
# st2.write(fname)
