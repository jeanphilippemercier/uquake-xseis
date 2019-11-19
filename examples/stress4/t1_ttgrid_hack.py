import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload

# import time
# from xseis2 import xutil
# from xseis2 import xchange
# # from xseis2 import xplot
# from xseis2 import xio
# # from spp.core.settings import settings
# from glob import glob

# from xseis2.h5stream import H5Stream

# from sqlalchemy import MetaData
# from sqlalchemy.sql import select
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# from sqlalchemy import create_engine
# # from xseis2.xsql import VelChange, XCorr, ChanPair, Base
# # from xseis2 import xchange_workflow as flow

# from microquake.core.helpers.timescale_db import get_continuous_data


# import redis

# import xseis2
# xseis2.__file__

# import logging
# import os

# from django.conf import settings

# from microquake.core.data.grid import read_grid
# from inventory.models import Site, Sensor
from microquake.core.settings import settings
from microquake.core.data.grid import read_grid

import logging
import os
# import threading

# from django.conf import settings

from microquake.core.data.grid import read_grid
from inventory.models import Site, Sensor

# from microquake.core.data.ttable import H5TTable
from microquake.core.data import GridData

from microquake.core.helpers.hdf5 import get_ttable_h5
logger = logging.getLogger(__name__)

tt = get_ttable_h5()

settings.nll_base

# Global variables used as cached lists
GRIDS_DATA = None
SENSORS = None
SITES_DICT = {}


def load_travel_time_grid(station, station_loc, phase):
    f_tt = os.path.join(settings.TRAVEL_TIME_FOLDER,
                        'OT.%s.%s.time.buf' % (phase.upper(), station))
    if not os.path.isfile(f_tt):
        raise ValueError("No travel time grid file found at {}".format(f_tt))
    logger.debug("starting reading grid file: %s" % f_tt)
    tt_grid = read_grid(f_tt, format='NLLOC')
    tt_grid.seed = station_loc
    logger.debug("finished reading grid file: %s" % f_tt)
    return tt_grid


def load_travel_time_grids(stations):
    # from pandas import DataFrame
    logger.info("Loading travel time grids")
    out_dict = {'station_id': [], 'phase': [], 'grid': []}

    for station in stations:
        try:
            for phase in ['P', 'S']:
                st_id = station.code
                # st_loc = [station.location_x, station.location_y, station.location_z]
                st_loc = station.loc
                grid = load_travel_time_grid(str(st_id), st_loc, phase)
                out_dict['phase'].append(phase)
                out_dict['grid'].append(grid)
            out_dict['station_id'].append(station.code)
        except ValueError as e:
            logger.warn("Error loading travel time: {}".format(e))
    logger.info("Loaded travel time grids")

    return out_dict


def load_travel_time_grids_from_h5(stations):

    # from pandas import DataFrame
    logger.info("Loading travel time grids")
    out_dict = {'station_id': [], 'phase': [], 'grid': []}
    htt = get_ttable_h5()

    for station in stations:
        # station = stations[0]
        if station.code not in htt.stations:
            continue

        row_ix = htt.index_sta(station.code)
        st_loc = htt.locations[row_ix]

        for phase, phase_key in zip(['P', 'S'], ['ttp', 'tts']):
            # print(key, phase)
            tt = htt.hf[phase_key][row_ix].reshape(htt.shape)

            grid = GridData(tt, spacing=htt.spacing, origin=htt.origin,
                        seed=st_loc, seed_label=station.code,
                        grid_type='VELOCITY')
            out_dict['phase'].append(phase)
            out_dict['grid'].append(grid)
            out_dict['station_id'].append(station.code)
    logger.info("Loaded travel time grids")

    return out_dict



nll_tts_dir = os.path.join(settings.nll_base, 'time')
settings.TRAVEL_TIME_FOLDER = nll_tts_dir

phase = 'P'
stations = settings.inventory.stations()
station = stations[0]
st_id = station.code
st_loc = station.loc
grid = load_travel_time_grid(str(st_id), st_loc, phase)


od = load_travel_time_grids(stations[:2])

od2 = load_travel_time_grids_from_h5(stations[:2])


# np.allclose(od['grid'][1].data, od2['grid'][1].data)










htt = get_ttable_h5()
tables = {'ttp': htt.hf['ttp'][:], 'tts': htt.hf['tts'][:]}

# from pandas import DataFrame
logger.info("Loading travel time grids")
out_dict = {'station_id': [], 'phase': [], 'grid': []}


for station in stations[:2]:
    # station = stations[0]
    if station.code not in htt.stations:
        continue

    row_ix = htt.index_sta(station.code)
    st_loc = htt.locations[row_ix]

    for phase, phase_key in zip(['P', 'S'], ['ttp', 'tts']):
        print(key, phase)

        tt = tables[phase_key][row_ix].reshape(htt.shape)

        grid = GridData(tt, spacing=htt.spacing, origin=htt.origin,
                    seed=st_loc, seed_label=station.code,
                    grid_type='VELOCITY')
        out_dict['phase'].append(phase)
        out_dict['grid'].append(grid)
        out_dict['station_id'].append(station.code)
logger.info("Loaded travel time grids")

return out_dict

# tt_s = htt.hf["tts"][row_ix].reshape(htt.shape)


# grid2 = GridData(data, spacing=1, origin=None,
#                  seed_label=None, seed=None, grid_type='VELOCITY',
#                  resource_id=None)


    for phase in ['P', 'S']:
        st_id = station.code
        # st_loc = [station.location_x, station.location_y, station.location_z]
        st_loc = station.loc
        grid = load_travel_time_grid(str(st_id), st_loc, phase)
        out_dict['phase'].append(phase)
        out_dict['grid'].append(grid)
    out_dict['station_id'].append(station.code)
except ValueError as e:
    logger.warn("Error loading travel time: {}".format(e))
logger.info("Loaded travel time grids")

return out_dict




