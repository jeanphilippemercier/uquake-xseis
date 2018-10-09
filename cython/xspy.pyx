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

	void CythonTest(vector[uint16_ptr]& vals, int ncol)  


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



# def GetRowPointers(np.ndarray[np.uint16_t, ndim=2] data):
# 	nrow = data.shape[0]
# 	cdef np.ndarray[np.int64_t, ndim=1] pointers = np.empty(nrow, dtype=np.int64)

# 	for i in range(nrow):
# 		pointers[i] = <uintptr_t>(&data[i, 0])

# 	return pointers


# def test0(np.ndarray[np.int64_t, ndim=1] ptrs, ncol):

# 	# cdef vector[uint16_t*] vect
# 	# cdef vector[int*] vect
# 	cdef vector[uint16_ptr] vect

# 	for i in range(ptrs.shape[0]):
# 		# vect.push_back(<uint16_t*>ptrs[i])
# 		vect.push_back(<uint16_t*>ptrs[i])

# 	# cdef uint16_ptr x
# 	# for x in vect:
# 	# 	# print("{0:x}".format(<unsigned long>x))
# 	# 	# print("{0:x}".format(<unsigned long>(x)[0]))
# 	# 	print("{0}".format(x[0]))
# 	# 	# print(x)

# 	CythonTest(vect, ncol)

	# for i in range(5):
	# 	# print(<uintptr_t>(vect[i]))
	# 	print("{0:x}".format(<unsigned long>vect[i]))


	# cdef vector[uint16_t*] vect
	# cdef int i, x
	# for i in range(ptrs.shape[0]):
	# 	vect.push_back(<uint16_t*>ptrs[i])
	# CythonTest(vect, ncol)


# def pySearchOnePhase2(np.ndarray[np.float32_t, ndim=2] data,
# 					sr,
# 					np.ndarray[np.float32_t, ndim=2] stalocs,
# 					np.ndarray[np.uint16_t, ndim=1] chanmap,
# 					np.ndarray[np.uint16_t, ndim=2] ttable,
# 					nthreads,
# 					outfile,
# 					debug
# 			   ):
# 	assert(data.shape[0] == chanmap.shape[0])
# 	assert(ttable.shape[0] == stalocs.shape[0])

# 	cdef string outfile_str = outfile.encode('UTF-8')
# 	cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(3, dtype=np.int64)
# 	# print("{0:x}".format(<unsigned long>&stalocs[0, 0]))

# 	SearchOnePhase(&data[0, 0], data.shape[0], data.shape[1],
# 					sr,
# 					&stalocs[0, 0], stalocs.shape[0],
# 					&chanmap[0],					
# 					&ttable[0, 0], ttable.shape[1],
# 					&out[0],
# 					nthreads,
# 					outfile_str,
# 					debug
# 					)
# 	return out








# def pySearchOnePhase(np.ndarray[np.float32_t, ndim=2] data,
# 					sr,
# 					np.ndarray[np.float32_t, ndim=2] stalocs,
# 					np.ndarray[np.uint16_t, ndim=1] chanmap,
# 					np.ndarray[np.uint16_t, ndim=2] ttable,					
# 					np.ndarray[np.int64_t, ndim=1] outbuf,					
# 					nthreads,
# 					outfile,
# 					debug
# 			   ):
# 	assert(data.shape[0] == chanmap.shape[0])
# 	assert(ttable.shape[0] == stalocs.shape[0])
# 	# assert(ttable.shape[1] == output.shape[0])
# 	# assert(ttable.shape[0] == ttable2.shape[0])
# 	# assert(ttable.shape[1] == ttable2.shape[1])

# 	cdef string outfile_str = outfile.encode('UTF-8')

# 	# ngrid = ttable1.shape[1]
# 	# print(ttable1.shape)
# 	# print(ttable2.shape)
# 	# print(ngrid)
# 	return SearchOnePhase(&data[0, 0], data.shape[0], data.shape[1],
# 					sr,
# 					&stalocs[0, 0], stalocs.shape[0],
# 					&chanmap[0],					
# 					&ttable[0, 0], ttable.shape[1],
# 					&outbuf[0],
# 					nthreads,
# 					outfile_str,
# 					debug
# 					)


# from libcpp.vector cimport vector



# cdef extern from "xseis2/workflows.h" namespace "xseis":

# 	void SearchWinDec2X(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_ptr, uint32_t ngrid, uint32_t* outbuf, uint32_t nthreads, string& file_out, int debug) 


# def fSearchWinDec2X(np.ndarray[np.float32_t, ndim=2] data,
# 					sr,
# 					np.ndarray[np.float32_t, ndim=2] stalocs,
# 					np.ndarray[np.uint16_t, ndim=1] chanmap,
# 					np.ndarray[np.uint16_t, ndim=2] ttable,					
# 					np.ndarray[np.uint32_t, ndim=1] outbuf,					
# 					nthreads,
# 					outfile,
# 					debug
# 			   ):
# 	assert(data.shape[0] == chanmap.shape[0])
# 	assert(ttable.shape[0] == stalocs.shape[0])
# 	# assert(ttable.shape[1] == output.shape[0])
# 	# assert(ttable.shape[0] == ttable2.shape[0])
# 	# assert(ttable.shape[1] == ttable2.shape[1])

# 	cdef string outfile_str = outfile.encode('UTF-8')

# 	# ngrid = ttable1.shape[1]
# 	# print(ttable1.shape)
# 	# print(ttable2.shape)
# 	# print(ngrid)
# 	return SearchWinDec2X(&data[0, 0], data.shape[0], data.shape[1],
# 					sr,
# 					&stalocs[0, 0], stalocs.shape[0],
# 					&chanmap[0],					
# 					&ttable[0, 0], ttable.shape[1],
# 					&outbuf[0],
# 					nthreads,
# 					outfile_str,
# 					debug
# 					)




	# void CorrSearchDec2X(float* rdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_p, uint32_t ngrid, uint32_t* outbuf, float* grid_p, uint32_t nthreads)

	# void CorrSearchDec2XBoth(float* rdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_p, uint16_t* ttable_s, uint32_t ngrid, uint32_t* outbuf, float* grid_p, uint32_t nthreads, string& logdir, int debug)  


# def fCorrSearchDec2X(np.ndarray[np.float32_t, ndim=2] data,
# 					sr,
# 					np.ndarray[np.float32_t, ndim=2] stalocs,
# 					np.ndarray[np.uint16_t, ndim=1] chanmap,
# 					np.ndarray[np.uint16_t, ndim=2] ttable,
# 					np.ndarray[np.uint32_t, ndim=1] outbuf,
# 					np.ndarray[np.float32_t, ndim=1] output,
# 					nthreads
# 			   ):
# 	assert(data.shape[0] == chanmap.shape[0])
# 	assert(ttable.shape[0] == stalocs.shape[0])
# 	assert(ttable.shape[1] == output.shape[0])

# 	return CorrSearchDec2X(&data[0, 0], data.shape[0], data.shape[1],
# 					sr,
# 					&stalocs[0, 0], stalocs.shape[0],
# 					&chanmap[0],					
# 					&ttable[0, 0], ttable.shape[1],
# 					&outbuf[0],
# 					&output[0],
# 					nthreads
# 					)


# cdef extern from "xseis/structures.h":
# 	ctypedef struct Arr_f "Array2D<float>":
# 		float *data_
# 		size_t nrow_
# 		size_t ncol_		
# 	ctypedef struct Arr_ui16 "Array2D<uint16_t>":
# 		unsigned short *data_
# 		size_t nrow_
# 		size_t ncol_
# 	ctypedef struct Vec_f "Vector<float>":
# 		float *data_
# 		size_t size_
# 	ctypedef struct Vec_ui16 "Vector<uint16_t>":
# 		unsigned short *data_
# 		size_t size_


# cdef extern from "xseis/interloc.h" namespace "interloc":

# 	void Search(Arr_f& data, Arr_f& stalocs, Vec_ui16& chanmap, int* tmeta, Arr_ui16& ttable, float *outbuf)

# 	void CorrSearchDec2X(Arr_f& data, float sr, Arr_f& stalocs, Vec_ui16& chanmap, Arr_ui16& ttable, uint32_t *outbuf, Vec_f& output, uint32_t nthreads)



# def fCorrSearchDec2X(np.ndarray[np.float32_t, ndim=2] data,
# 					sr,
# 					np.ndarray[np.float32_t, ndim=2] stalocs,
# 					np.ndarray[np.uint16_t, ndim=1] chanmap,
# 					np.ndarray[np.uint16_t, ndim=2] ttable,
# 					np.ndarray[np.uint32_t, ndim=1] outbuf,
# 					np.ndarray[np.float32_t, ndim=1] output,
# 					nthreads
# 			   ):

# 	return CorrSearchDec2X(Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
# 					sr,
# 					Arr_f(&stalocs[0, 0], stalocs.shape[0], stalocs.shape[1]),
# 					Vec_ui16(&chanmap[0], chanmap.shape[0]),					
# 					Arr_ui16(&ttable[0, 0], ttable.shape[0], ttable.shape[1]),
# 					&outbuf[0],
# 					Vec_f(&output[0], output.shape[0]),
# 					nthreads
# 					)



# def fSearch(np.ndarray[np.float32_t, ndim=2] data,
# 			  np.ndarray[np.float32_t, ndim=2] stalocs,
# 			  np.ndarray[np.uint16_t, ndim=1] chanmap,
# 			  np.ndarray[int, ndim=1] tmeta,
# 			  np.ndarray[np.uint16_t, ndim=2] ttable,
# 			  np.ndarray[np.float32_t, ndim=1] outbuf,
# 			   ):

# 	return Search(Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
# 					Arr_f(&stalocs[0, 0], stalocs.shape[0], stalocs.shape[1]),
# 					Vec_ui16(&chanmap[0], chanmap.shape[0]),
# 					&tmeta[0],
# 					Arr_ui16(&ttable[0, 0], ttable.shape[0], ttable.shape[1]),
# 					&outbuf[0],
# 					)

