# import eikonal.data as data
import numpy as np
import matplotlib.pyplot as plt
import h5py
from xseis import xplot
from xseis import xutil

from importlib import reload

plt.ion()

ddir = '/home/phil/data/oyu/'
# syncdir = '/home/phil/data/oyu/results/'
DIR_TTS = ddir + 'NLLOC_grids/'

sr = 6000.

ttP, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.P")
# ttP = (ttP * dsr).astype(np.uint16)
# ttS, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.S")
# ttS = (ttS * dsr).astype(np.uint16)

stalocs = xutil.shift_locs_ot(slocs)
gridlocs = xutil.shift_locs_ot(xutil.gdef_to_points(gdef))

reload(xplot)
# xplot.plot3d(stalocs, gridlocs[::1000])
src_loc = np.array([1600, 1400, 1000])
# xplot.plot3d(stalocs, src_loc)
dd = xutil.dist2many(src_loc, gridlocs)
print(np.min(dd))
isrc = np.argmin(dd)

d2src = xutil.dist2many(src_loc, stalocs)

vp, vs = 5000., 3000.

tt0_src = (d2src / vp * sr + 0.5).astype(int)
tt1_src = (d2src / vs * sr + 0.5).astype(int)

names = np.array(list(ndict.keys()))
chanmap = np.repeat(np.arange(len(names)), 3)

exdat = np.load(ddir + "event_waveforms.npy")
# xplot.sigs(exdat, labels=np.arange(exdat.shape[0]))
# sig1 = exdat[28][2800:3000]
# sig2 = exdat[28][3550:3750]
sigs = exdat[93:96]
scale = np.max(sigs)
# xplot.sigs(sigs)
pwave = sigs[:, 3180: 3480] / scale
swave = sigs[:, 4220: 4520] / scale
xutil.taper2d(pwave, 50)
xutil.taper2d(swave, 50)
# xplot.sigs(swave)
# plt.plot(sig1)
# plt.plot(sig2)

npts = 6000
nchan = len(chanmap)
dsim = np.zeros((nchan, npts), dtype=np.float32)
cgroups = xutil.chan_groups(chanmap)
wlen = pwave.shape[1]
ixot = 1000

for i, group in enumerate(cgroups):
	atten = 1. / xutil.dist(src_loc, stalocs[i])
	for comp, ichan in enumerate(group):
		ix = ixot + tt0_src[i]
		dsim[ichan, ix: ix + wlen] += pwave[comp] * atten
		ix = ixot + tt1_src[i]
		dsim[ichan, ix: ix + wlen] += swave[comp] * atten

band = [10, 1000]
xutil.add_noise(dsim, band, sr, power=0.5e-2)
# xplot.sigs(dsim[::10])

dd = xutil.dist2many(src_loc, stalocs)[chanmap]
isort = np.argsort(dd)
# xplot.sigs(dsim[isort])
# xplot.im(dsim[isort])
# mx = np.max(dsim, axis=1)
# plt.scatter(mx, dd)

dec = 4
dsr = sr / dec
xutil.bandpass(dsim, [50, dsr / 2], sr)
dat = dsim[:, ::dec]
xplot.im(dat[isort])


ddir = "/home/phil/data/oyu/synthetic/"
hf = h5py.File(ddir + 'sim_p5s3.h5', 'w')
hf.create_dataset('data', data=dat.astype(np.float32))
hf.create_dataset('src_loc', data=src_loc.astype(np.float32))
# hf.create_dataset('src_time', data=src_time.astype(np.float32))
# print(list(hf.keys()))
hf.create_dataset('sta_locs', data=stalocs.astype(np.float32))
hf.create_dataset('chan_map', data=chanmap.astype(np.uint16))
hf.attrs['samplerate'] = float(dsr)
hf.close()


from obspy import Stream, Trace
comps = ["X", "Y", "Z"]
st = Stream()
for i, group in enumerate(cgroups):
	sta = names[i]
	for comp, ichan in enumerate(group):
		tr = Trace(data=dsim[ichan])
		tr.stats.sampling_rate = sr
		tr.stats.channel = comps[comp]
		tr.stats.station = names[i]
		st.append(tr)

fle = ddir + "sim_dat.mseed"
st.write(fle, format='MSEED', reclen=4096)

plt.plot(st[isort[-1]].data)
plt.plot(st[isort[0]].data)

for i, sig in enumerate(dsim):
	tr = Trace(data=sig)
	tr.stats.sampling_rate = sr
	tr.stats.channel
	st.append(tr)


# # ttP2 = ((ttP * sr).astype(np.uint16) / sr).astype(np.float32)
# stalocs = xutil.shift_locs_ot(slocs)
# gridlocs = xutil.shift_locs_ot(xutil.gdef_to_points(gdef))

# hf = h5py.File(ddir + 'nll_ttable.h5', 'w')
# hf.create_dataset('sta_locs', data=stalocs.astype(np.float32))
# hf.create_dataset('grid_locs', data=gridlocs.astype(np.float32))
# hf.create_dataset('tts_p', data=ttP)
# hf.create_dataset('tts_s', data=ttS)
# hf.create_dataset('grid_def', data=gdef)
# hf.attrs['samplerate'] = float(sr)
# hf.close()
















