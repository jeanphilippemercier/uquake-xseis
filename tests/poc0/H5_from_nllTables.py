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

# sr = 6000.
sr = 3000.
# ttP, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.P")

ttP, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.P")
ttP = (ttP * sr).astype(np.uint16)
ttS, slocs, ndict, gdef = xutil.ttable_from_nll_grids(DIR_TTS, key="OT.S")
ttS = (ttS * sr).astype(np.uint16)

# stalocs = xutil.shift_locs_ot(slocs)
# gridlocs = xutil.shift_locs_ot(xutil.gdef_to_points(gdef))
gridlocs = xutil.gdef_to_points(gdef)
stalocs = slocs
names = np.array(list(ndict.keys()), dtype='S4')


hf = h5py.File(ddir + 'synthetic/nll_ttable.h5', 'w')
hf.create_dataset('sta_locs', data=stalocs.astype(np.float32))
hf.create_dataset('grid_locs', data=gridlocs.astype(np.float32))
hf.create_dataset('tts_p', data=ttP)
hf.create_dataset('tts_s', data=ttS)
hf.create_dataset('grid_def', data=gdef)
hf.create_dataset('sta_names', data=names)
hf.attrs['samplerate'] = float(sr)
hf.close()








# shape, origin, spacing = gdef[:3], gdef[3:6], float(gdef[6])
# maxes = origin + shape * spacing
# maxes = xutil.shift_locs_ot(maxes)
# mins = xutil.shift_locs_ot(origin)
# # maxes = mins + shape * spacing
# inter = np.ravel(np.column_stack((mins, maxes)))
# glims = np.array([mins[0], maxes[0], mins[1]])
# glims = np.hstack((og, maxes))
# glims = np.concatenate((og, maxes, [spacing])).astype(np.float32)











