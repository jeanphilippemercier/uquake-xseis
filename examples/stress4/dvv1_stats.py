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

plt.ion()

# filename = os.path.join(os.environ['SPP_COMMON'], "stations.pickle")
# with open(filename, 'rb') as f:
#     stations = pickle.load(f)
###############################

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
