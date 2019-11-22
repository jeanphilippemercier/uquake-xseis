from importlib import reload
import os
import numpy as np
import os
import time
# import h5py
# from glob import glob
from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
import matplotlib.pyplot as plt
import pickle
from microquake.core import read

plt.ion()

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations = pickle.load(f)
###############################

sr_raw = 6000.0
dsr = 1000.0
# whiten_freqs = np.array([60, 80, 320, 350])
fband = [50, dsr / 2]
decf = int(sr_raw / dsr)


fname = os.path.join(os.environ['SPP_COMMON'], "cont10min.mseed")
stream = read(fname)
stream.sort()
req_start_time = stream[0].stats.starttime

tr = stream[0]
tr.data += np.linspace(0, np.max(tr.data), len(tr.data))
tr.data[:10000] = np.nan
tr.data[50000:1000000] = np.nan
plt.plot(tr.data)

nchan = len(stream)
nsamp_raw = np.max([len(tr.data) for tr in stream])
nsamp = nsamp_raw // decf
data = np.zeros((nchan, nsamp), dtype=np.float32)

chan_names = []

for i, tr in enumerate(stream):
    print(tr)
    tr.data -= np.nanmean(tr.data)
    tr.data = linear_detrend_nan(tr.data)
    tr.data = np.nan_to_num(tr.data)
    tr.filter('bandpass', freqmin=fband[0], freqmax=fband[1])
    # sig = np.sign(tr.data[::decf])
    sig = tr.data[::decf]
    i0 = int((tr.stats.starttime - req_start_time) * dsr + 0.5)
    slen = min(len(sig), nsamp - i0)
    data[i, i0: i0 + slen] = sig[:slen]
    chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

chan_names = np.array(chan_names)
# data = np.sign(data).astype(np.bool_)

plt.plot(data[0])



tr = tr
sig = tr.data
sr = tr.stats.sampling_rate
plt.plot(sig)

xplot.freq(sig, sr)

for tr in st:

    data = tr.data
    nans, x = xutil.nan_helper(data)
    number_of_nans = data[nans].size
    iz = np.where(data == 0)[0]
    print(number_of_nans, len(iz))


plt.plot(st[2].data)

np.nonzero(sig)

data = st[-1].data

nans, x = xutil.nan_helper(data)
number_of_nans = data[nans].size
iz = np.where(data == 0)[0]
print(number_of_nans, len(iz))
plt.plot(data, alpha=0.2)
xv = np.arange(len(data))
plt.scatter(xv[iz], data[iz], color='red')

# plt.plot(iz)



import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def linear_detrend_nan(y):
    x = np.arange(len(y))
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind], y[not_nan_ind])
    detrend_y = y - (m * x + b)
    return detrend_y


# create data
x = np.linspace(0, 2 * np.pi, 500)
y = np.random.normal(0.3 * x, np.random.rand(len(x)))
drops = np.random.rand(len(x))
y[drops > .95] = np.nan
# plt.plot(x, y)


xv = np.arange(len(y))
detrend_y = linear_detrend_nan(y)
plt.plot(xv, detrend_y)
detrend_y = np.nan_to_num(detrend_y)
plt.plot(xv, detrend_y)











x, /, out, where, casting, order, dtype, subok, signature, extobj
sr = 1000.0
keeplag_sec = 1.0

reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05

dist = 200
coda_start_sec = dist / coda_start_vel

fband_sig = np.array([50, 80, 280, 300])
fband_noise = np.array([50, 80, 280, 300])
whiten_freqs = fband_sig
tt_change_percent = 0.03
# tt_change_percent = 0.3
# noise_scale = 1.0
noise_scale = 0.0
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None

nsamp = int(keeplag_sec * sr)

sig1, sig2 = xchange.mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
# plt.plot(sig1)
# plt.plot(sig2)

# np.corrcoef(sig1, sig2)
reload(xutil)
reload(xchange)
vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)

reload(xchange)
xchange.plot_dvv(vals, dvv_true=tt_change_percent)
xplot.quicksave()


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
tt_change_percent = 0.03
# noise_scale = 1.0
noise_scale = 0.5
dvv_outlier_clip = 1.0
# dvv_outlier_clip = None

out = []
niter = 500
for i in range(niter):
    print(f"{i} / {niter}")

    sig1, sig2 = mock_corrs_dvv(nsamp, sr, tt_change_percent, fband_sig, fband_noise, noise_scale)
    vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    out.append(vals)

dvv = np.array([v["dvv"] for v in out])
error = [v["regress"][2] for v in out]
coeffs = [v["coeff"] for v in out]


import matplotlib.gridspec as gridspec
reload(xplot)

fig = plt.figure(figsize=(12, 8), facecolor='white')
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0, :])

plt.hist(dvv, bins=70)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
plt.title(f"dvv | {niter} iters | noise_scale {noise_scale} | outlier_clip {dvv_outlier_clip} | sig {fband_sig}Hz | noise {fband_noise} Hz")
plt.xlabel("dvv measurement percent")
plt.ylabel("count")
plt.xlim(np.array([-0.05, 0.05]) + tt_change_percent)
plt.legend()

ax = fig.add_subplot(gs[1, 0])
diff = np.abs(dvv - tt_change_percent)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
# plt.scatter(diff, error, s=8, alpha=0.4)
plt.scatter(dvv, error, s=8, alpha=0.4)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("linear_fit_error")

ax = fig.add_subplot(gs[1, 1])
diff = np.abs(dvv - tt_change_percent)
# plt.scatter(diff, coeffs, s=8, alpha=0.4)
plt.scatter(dvv, coeffs, s=8, alpha=0.4)
plt.axvline(tt_change_percent, color='red', label='dvv true', linestyle='--', alpha=0.5)
plt.xlabel("abs(dvv_meas - dvv_true)")
plt.ylabel("corr_coeff")
plt.tight_layout()
xplot.quicksave()
