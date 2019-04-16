# distutils: language = c++
# cython: language_level=3
# Cython interface file for wrapping the object
cimport numpy as np
import numpy as np
from libcpp cimport bool
from libcpp.string cimport string
from libc.stdint cimport uint16_t, uint32_t, int64_t
from libc.stdint cimport uintptr_t
# from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

ctypedef uint16_t* uint16_ptr # pointer workaround

cdef extern from "xseis2/workflows.h" namespace "xs":

	void SearchOnePhase(
		float* raw_data_ptr, uint32_t nchan, uint32_t nsample, # ptr to raw data (size = nchan x nsamp)
		float samplerate,
		float* station_locations_ptr, uint32_t nsta, # ptr to station locs (size = nsta x 3)
		uint16_t* channel_map_ptr, # ptr to channel map (size = nchan)
		vector[uint16_ptr]& ttable_row_ptrs, uint32_t ngrid,
		int64_t* outbuf,
		vector[float]& whiten_corner_freqs, # 4 corner frequencies to whiten between
		float pair_dist_min, # uses only correlation pairs between dist_min and dist_max
		float pair_dist_max,
		float cc_smooth_length_sec, # length to absolute value smooth ccfs before beamforming
		uint32_t nthreads,
		int debug_lvl,
		string& debug_file
		)


def pySearchOnePhase(np.ndarray[np.float32_t, ndim=2] data,
					samplerate,
					np.ndarray[np.uint16_t, ndim=1] chanmap,
					np.ndarray[np.float32_t, ndim=2] stalocs,
					np.ndarray[np.int64_t, ndim=1] tt_ptrs,
					ngrid,
					np.ndarray[np.float32_t, ndim=1] whiten_corner_freqs,
					pair_dist_min,
					pair_dist_max,
					cc_smooth_length_sec,
					nthreads,
					debug_lvl,
					debug_file
			   ):

# Pre-process, cross-correlate and beamform raw_data using provided traveltime table.
# The provided traveltimes determine which phase (e.g P,S) is searched for.

# :param data: 2d array of channel data (1 row per channel), assumes all starttimes equal
# :param samplerate: sampling rate of data
# :param chanmap: indexes to map each channel to its respective station location
# :param stalocs: unique station locations (shape = nsta x 3)
# :param tt_ptrs: pointers to rows in the traveltime table corresponding to stalocs
# :param ngrid: number of columns (or search locations) in the traveltime tables
# :param whiten_corner_freqs: array of 4 floats defining whitening corner frequencies in Hz
# :param pair_dist_min: uses only correlation pairs between dist_min and dist_max (meters)
# :param pair_dist_max:
# :param cc_smooth_length_sec: length to absolute value smooth ccfs before beamforming
# :param debug_lvl: int 0 (disabled) 1 (logging) 2 (logging + data dump) to debug_file


# * Important that rows of tt_ptrs are consistent with rows of station locations
# * The chanmap is used to determine the traveltimes to use for each channel
#   in the data array.


	chanmap = np.ascontiguousarray(chanmap)
	stalocs = np.ascontiguousarray(stalocs)
	data = np.ascontiguousarray(data)

	assert(data.shape[0] == chanmap.shape[0])
	assert(tt_ptrs.shape[0] == stalocs.shape[0])

	# convert np array to std::vector
	cdef vector[float] corner_freqs_vec
	for i in range(whiten_corner_freqs.shape[0]):
		corner_freqs_vec.push_back(<float>whiten_corner_freqs[i])

	# buil vec of c-pointers from casted python ptrs
	cdef vector[uint16_ptr] tt_ptrs_vec
	for i in range(tt_ptrs.shape[0]):
		tt_ptrs_vec.push_back(<uint16_t*>tt_ptrs[i])

	cdef string debug_file_str = debug_file.encode('UTF-8')
	cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(3, dtype=np.int64)
	# print("{0:x}".format(<unsigned long>&stalocs[0, 0]))

	SearchOnePhase(&data[0, 0], data.shape[0], data.shape[1],
					samplerate,
					&stalocs[0, 0], stalocs.shape[0],
					&chanmap[0],
					tt_ptrs_vec, ngrid,
					&out[0],
					corner_freqs_vec,
					pair_dist_min,
					pair_dist_max,
					cc_smooth_length_sec,
					nthreads,
					debug_lvl,
					debug_file_str
					)

	out_list = [float(out[0]) / 10000., out[1], out[2]]
	return out_list


