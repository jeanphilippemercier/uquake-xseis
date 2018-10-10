# distutils: language = c++
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

	void SearchOnePhase(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, vector[uint16_ptr]& tt_ptrs_vec, uint32_t ngrid, int64_t* outbuf, uint32_t nthreads, int debug, string& file_out)


def pySearchOnePhase(np.ndarray[np.float32_t, ndim=2] data,
					sr,
					np.ndarray[np.uint16_t, ndim=1] chanmap,
					np.ndarray[np.float32_t, ndim=2] stalocs,					
					np.ndarray[np.int64_t, ndim=1] tt_ptrs,
					ngrid,
					nthreads,
					debug,
					outfile
			   ):

	stalocs = np.ascontiguousarray(stalocs)
	data = np.ascontiguousarray(data)

	assert(data.shape[0] == chanmap.shape[0])
	assert(tt_ptrs.shape[0] == stalocs.shape[0])

	# buil vec of c-pointers from casted python ptrs
	cdef vector[uint16_ptr] tt_ptrs_vec
	for i in range(tt_ptrs.shape[0]):
		tt_ptrs_vec.push_back(<uint16_t*>tt_ptrs[i])

	cdef string outfile_str = outfile.encode('UTF-8')
	cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(3, dtype=np.int64)
	# print("{0:x}".format(<unsigned long>&stalocs[0, 0]))

	SearchOnePhase(&data[0, 0], data.shape[0], data.shape[1],
					sr,
					&stalocs[0, 0], stalocs.shape[0],
					&chanmap[0],					
					tt_ptrs_vec, ngrid,
					&out[0],
					nthreads,
					debug,
					outfile_str
					)

	out_list = [float(out[0]) / 10000., out[1], out[2]]
	return out_list


