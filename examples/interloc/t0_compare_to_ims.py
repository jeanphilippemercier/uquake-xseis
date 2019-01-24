from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import glob
from microquake.core import read
from xseis2 import xspy
# from xseis2 import xflow
# from xseis2 import xutil
from xseis2 import xplot
from spp.utils.application import Application
from microquake.core.util import tools
from microquake.core import read_events

plt.ion()

app = Application()
# mseed_file = os.path.join(app.common_dir, "synthetic", "sim_dat_noise.mseed")

# basename = "2019-01-07T20_39_25.549443Z"
basename = "2019-01-08T04_50_06.392876Z"

mseed_file = os.path.join(app.common_dir, "real2", basename + ".mseed")
cat_file = os.path.join(app.common_dir, "real2", basename + ".xml")
# cat_file = os.path.join(app.common_dir, "real2", "2019-01-07T20_39_25.549443Z.xml")

ev = read_events(cat_file)[0]
true_loc = ev.origins[0].loc

nthreads = int(4)
debug = int(2)
# debug = int(0)
dsr = float(3000.)
wlen_sec = float(2.0)
# wlen_sec = float(0.8)

# app.write_ttable_h5()
htt = app.get_ttable_h5()
stalocs = htt.locations
ttable = (htt.hf['ttp'][:] * dsr).astype(np.uint16)
# ttable = (htt.hf['tts'][:] * dsr).astype(np.uint16)
ngrid = ttable.shape[1]
tt_ptrs = np.array([row.__array_interface__['data'][0] for row in ttable])

# reload(xutil)
# reload(xflow)

st = read(mseed_file)

# st.zpad_names()
data, sr, t0 = st.as_array(wlen_sec)
data = tools.decimate(data, sr, int(sr / dsr))
data = np.nan_to_num(data)
chanmap = st.chanmap().astype(np.uint16)
ikeep = htt.index_sta(st.unique_stations())

# np.array([np.isnan(tr.data).any() for tr in st])

print(np.isnan(data).any())
# xplot.sigs(data)
# [len(tr) for tr in st]

npz_file = os.path.join(app.common_dir, "dump", "output.npz")
out = xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep], ngrid, nthreads, debug, npz_file)
vmax, imax, iot = out
print("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))
# lmax = xutil.imax_to_xyz_gdef(imax, gdef)
lmax = htt.icol_to_xyz(imax)
print(lmax.astype(int))
otime = t0 + iot / dsr
ot_epoch = (otime.datetime - datetime(1970, 1, 1)) / timedelta(seconds=1)
# true_loc = np.array([651600, 4767420, 200])
# print('correct loc: ', np.allclose(lmax, true_loc))

print("location diff (m): ", true_loc - lmax)
print("location diff (m): ", np.linalg.norm(true_loc - lmax))

l1 = np.array([652025., 4767445, -50])
l2 = np.array([652025., 4767470, -25])
print("location diff (m): ", l1 - l2)
print("location diff (m): ", np.linalg.norm(l1 - l2))

##########################################################


with np.load(npz_file, mmap_mode='r') as npz:
	print(npz.files)
	dat = npz['dat_preproc']
	ckeys = npz['sta_pairs']
	ccs = npz['dat_cc']
	wtt = npz['tts_src'].astype(int)
	gpower = npz['grid_power']
	droll = npz['dat_rolled']
	stack = npz['dat_stack']


xplot.im(droll, labels=['sample (samplerate=%d)' % dsr, 'channel'])
dstack = np.mean(droll, axis=0)
plt.plot(dstack / np.max(dstack) * droll.shape[0] / 5, color='red')
plt.title('Data rolled based on event')


# grad = np.gradient(dstack, 2)
# plt.plot(dstack)
# plt.plot(grad)

# plt.plot(np.grad)

from xseis2 import xplot3d
reload(xplot3d)
xplot3d.power(gpower, htt.shape, htt.origin, htt.spacing, stalocs, lmax=lmax)
# xplot3d.power(gpower, gdef, stalocs, lmax=lmax)
# xplot3d.power(gpower, htt.shape, htt.origin stalocs[ikeep], labels=stations)

# ttwin = (ttP_keep[:, out[1]]).astype(int)
# print(np.allclose(ttwin, wtt))











groups = xutil.chan_groups(chanmap)

picks = ot + wtt
shifts = np.arange(0, picks.shape[0], 1) * 1.2
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
xplot.chan_groups(dat, groups, shifts, alpha=0.5, zorder=0)

plt.axvline(ot, linestyle='--', color='red')
ax.scatter(picks, shifts, color='red', s=30, marker='|', zorder=1, label='homo')
plt.legend()


mxs = np.max(ccs, axis=1)
plt.plot(mxs)



# %time xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep], ngrid, nthreads, npz_file, debug)
# reload(xutil)
# reload(xplot3d)
# stalocs = np.ascontiguousarray(stalocs[ikeep])
# %timeit a = np.ascontiguousarray(data[::2])

# dist = 3050.0
# minvel = 3100.
# time = dist / minvel




