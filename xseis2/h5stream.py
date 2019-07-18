import numpy as np
from glob import glob
import os
import h5py
from datetime import timedelta, datetime


class H5Stream(object):
    """docstring for H5TTable"""

    def __init__(self, path, dset_key='data'):
        self.path = path
        self.hf = h5py.File(path, 'r')
        self.keys = list(self.hf.keys())
        self.set_dataset(dset_key)
        self.samplerate = self.hf.attrs['samplerate']
        self.channels = self.hf['channels'][:].astype(str)
        # self.stations = self.hf['stations'][:].astype('U4')
        self._nameixs = dict(zip(self.channels, np.arange(len(self.channels))))
        # self.starttime = UTCDateTime(self.hf.attrs['starttime'])
        self.starttime = datetime.fromtimestamp(self.hf.attrs['starttime'])
        self.endtime = self.starttime + timedelta(seconds=self.shape[1] / self.samplerate)
        # self.locations = self.hf['locations'][:]
        # self.coords = self.hf['grid_locs'][:]

    def set_dataset(self, key):
        if key in self.keys:
            self.dset = self.hf[key]
            self.shape = self.dset.shape
        else:
            raise KeyError('dataset %s does not exist' % key)

    def get_row_indexes(self, names):
        if isinstance(names, (list, np.ndarray)):
            return np.array([self._nameixs[name] for name in names])
        else:
            return self._nameixs[names]

    def query(self, channels, t0, t1):
        i0 = self.time_to_index(t0)
        i1 = self.time_to_index(t1)
        irows = self.get_row_indexes(channels)
        # nsamp = int((i1 - i0) * len(irows))
        # print(f"loading {i0}:{i1} for {len(irows)} rows ({nsamp} samples)")

        return self.dset[irows, i0:i1]

    def time_to_index(self, t):
        return int((t - self.starttime).total_seconds() * self.samplerate + 0.5)

    def slice_gen(self, starttime, endtime, chans, wlen, stepsize=None):

        if stepsize is None:
            stepsize = wlen

        wlen_td = timedelta(seconds=wlen)
        stepsize_td = timedelta(seconds=stepsize)
        t0 = starttime

        while (t0 <= endtime - wlen_td):
            t1 = t0 + stepsize_td
            print(t0, t1)
            dat = self.query(chans, t0, t1)
            t0 = t1
            yield dat

    def close(self):
        self.hf.close()
