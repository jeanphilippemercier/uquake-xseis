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
import pickle

plt.ion()


filename = os.path.join(os.environ['SPP_COMMON'], "stations.pickle")

with open(filename, 'rb') as f:
    stations = pickle.load(f)


dsr = 1000.0
# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cc_wlen_sec = 20.0
stepsize_sec = cc_wlen_sec
# stepsize_sec = cc_wlen_sec / 2
keeplag_sec = 1.0
stacklen_sec = 100.0
onebit = True
min_pair_dist = 100
max_pair_dist = 1000

####################################


def mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale):

    sig1 = xutil.noise1d(nsamp, fband_sig, sr, scale=1, taplen=0)
    sig2 = xchange.stretch(sig1, sr, tt_change_percent)

    sig1 = np.concatenate((sig1[::-1][1:], sig1))
    sig2 = np.concatenate((sig2[::-1][1:], sig2))
    nsamp = len(sig1)
    noise1 = xutil.noise1d(nsamp, fband_noise, sr, scale=noise_scale, taplen=0)
    noise2 = xutil.noise1d(nsamp, fband_noise, sr, scale=noise_scale, taplen=0)
    sig1 += noise1
    sig2 += noise2
    return sig1, sig2


coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
sr = dsr

dist = 500
coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 280, 300])
fband_noise = np.array([50, 80, 280, 300])
whiten_freqs = fband_sig
tt_change_percent = 0.01
noise_scale = 0.8
dvv_outlier_clip = 0.1

nsamp = int(keeplag_sec * sr)

sig1, sig2 = mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
# plt.plot(sig1)
# plt.plot(sig2)

# np.corrcoef(sig1, sig2)
coeff = xutil.pearson_coeff(sig1, sig2)
print(f"coeff: {coeff}")
reload(xutil)
reload(xchange)
dvv, error = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, plot=True)

#################################################


coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
sr = dsr

dist = 500
coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 280, 300])
fband_noise = np.array([50, 80, 280, 300])
whiten_freqs = fband_sig
tt_change_percent = 0.01
noise_scale = 1.5
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None

out = []
niter = 1000
for i in range(niter):
    print(f"{i} / {niter}")

    sig1, sig2 = mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
    coeff = xutil.pearson_coeff(sig1, sig2)
    # print(f"coeff: {coeff}")
    reload(xutil)
    reload(xchange)
    dvv, error = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip, plot=False)
    out.append([dvv, error, coeff])
dvv, error, coeffs = np.array(out).T

plt.hist(dvv, bins=70)
plt.axvline(tt_change_percent, color='red', label='dvv true')
plt.title(f"dvv | {niter} iters | noise_scale {noise_scale} | outlier_clip {dvv_outlier_clip} | sig {fband_sig}Hz | noise {fband_noise} Hz")
plt.xlabel("dvv measurement percent")
plt.ylabel("count")
plt.xlim(np.array([-0.05, 0.05]) + tt_change_percent)
plt.tight_layout()
plt.legend()


diff = np.abs(dvv - tt_change_percent)
plt.scatter(diff, error)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("linear_fit_error")
plt.tight_layout()


diff = np.abs(dvv - tt_change_percent)
plt.scatter(diff, coeffs)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("corr_coeff")
plt.tight_layout()
