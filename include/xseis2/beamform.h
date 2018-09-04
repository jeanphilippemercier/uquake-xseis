#pragma once

#include <omp.h>
#include "xseis2/core.h"
#include "xseis2/signal.h"
// #include <random>


namespace xseis {

inline float AngleBetweenPoints(float* a, float*b) 
{
	return std::atan((a[1] - b[1]) / (a[0] - b[0]));
	// return std::atan2(a[1] - b[1], a[0] - b[0]);
}

inline float DistCartesian(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	float dz = a[2] - b[2];
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline float DistCartesian2D(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	return std::sqrt(dx * dx + dy * dy);
}


inline float DistCartesian(gsl::span<float> a, gsl::span<float> b)
{	
	// float val = 0;
	float v[3];
	v[0] = a[0] - b[0];
	v[1] = a[1] - b[1];
	v[2] = a[2] - b[2];
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}


float DistDiff(float* a, float* b, float* c) {	
	return DistCartesian(a, c) - DistCartesian(b, c);
}


// Divide grid into blocks to prevent cache invalidations while writing to buffer from multiple threads
// Requires all rows are aligned and blocksize is a multiple of alignment 
void InterLocBlocks(const VecOfSpans<float> data_cc, const VecOfSpans<uint16_t> ckeys, const VecOfSpans<uint16_t> ttable, gsl::span<float> output, uint32_t blocksize=1024 * 8, float scale_pwr=1.0)
{	
	// note asserts incorrectly pass when called through cython with python owned memory
	assert((uintptr_t) data_cc[1].data() % MEM_ALIGNMENT == 0);
	assert((uintptr_t) output.data() % MEM_ALIGNMENT == 0); 
	assert((uintptr_t) ttable[1].data() % MEM_ALIGNMENT == 0);	
	assert(ckeys.size() == data_cc.size());	

	const uint16_t hlen = data_cc[0].size() / 2;	
	const size_t ncc = data_cc.size();
	const uint32_t ngrid = ttable[0].size();

	#pragma omp parallel for
	for(uint32_t iblock = 0; iblock < ngrid; iblock += blocksize) {

		float* out_ptr = &output[iblock];
		assert((uintptr_t) out_ptr % MEM_ALIGNMENT == 0);

		uint32_t blocklen = std::min(ngrid - iblock, blocksize);
		std::fill(out_ptr, out_ptr + blocklen, 0);
		
		// Migrate single ccf on to grid based on tt difference
		for (size_t i = 0; i < ncc; ++i) {				

			uint16_t* tts_sta1 = ttable[ckeys[i][0]].data() + iblock;	
			uint16_t* tts_sta2 = ttable[ckeys[i][1]].data() + iblock;	
			float* cc_ptr = data_cc[i].data();

			#pragma omp simd aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (size_t j = 0; j < blocklen; ++j) {
				out_ptr[j] += cc_ptr[hlen + tts_sta2[j] - tts_sta1[j]];
			}
		}
	}

	Multiply(output, scale_pwr / static_cast<float>(ncc));
	// float norm = scale_pwr / static_cast<float>(ncc);
	// for(size_t i = 0; i < output.size_; ++i) output[i] *= norm;
}



void InterLocBlocks2(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, Vector<float>& output, uint32_t blocksize=1024 * 8, float scale_pwr=100)
{
	// Divide grid into chunks to prevent cache invalidations during writing (see Ben Baker migrate)
	// This uses less memory but was a bit slower atleast in my typical grid/ccfs sizes
	// UPdate: When grid sizes >> nccfs and using more than 15 cores faster than InterLoc above

	// note these asserts dont work when called through cython (python owned memory)
	assert((uintptr_t) data_cc.row(1) % MEM_ALIGNMENT == 0);
	assert((uintptr_t) output.data() % MEM_ALIGNMENT == 0); 
	// assert((uintptr_t) ttable.row(1) % MEM_ALIGNMENT == 0);	

	// const size_t cclen = data_cc.ncol();
	const uint16_t hlen = data_cc.ncol() / 2;	
	const size_t ncc = data_cc.nrow();
	const uint32_t ngrid = ttable.ncol();
	uint32_t blocklen;

	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;
	float *out_ptr = nullptr;

	// printf("blocksize %lu, ngrid %lu \n", blocksize, ngrid);

	#pragma omp parallel for private(tts_sta1, tts_sta2, cc_ptr, out_ptr, blocklen)
	for(uint32_t iblock = 0; iblock < ngrid; iblock += blocksize) {

		blocklen = std::min(ngrid - iblock, blocksize);
		out_ptr = output.data_ + iblock;
		std::fill(out_ptr, out_ptr + blocklen, 0);
		
		for (size_t i = 0; i < ncc; ++i) {				

			tts_sta1 = ttable.row(ckeys(i, 0)) + iblock;	
			tts_sta2 = ttable.row(ckeys(i, 1)) + iblock;
			cc_ptr = data_cc.row(i);

			// Migrate single ccf on to grid based on tt difference
			// #pragma omp simd aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			#pragma omp simd aligned(out_ptr, cc_ptr: MEM_ALIGNMENT)			
			for (size_t j = 0; j < blocklen; ++j) {
				out_ptr[j] += cc_ptr[hlen + tts_sta2[j] - tts_sta1[j]];
			}
		}
	}

	Multiply(output.span(), scale_pwr / static_cast<float>(ncc));
	// float norm = scale_pwr / static_cast<float>(ncc);
	// for(size_t i = 0; i < output.size_; ++i) output[i] *= norm;
}



void FillTravelTimeTable(Array2D<float>& locs1, Array2D<float>& locs2, float vel, float sr, Array2D<uint16_t>& ttable)
{
	float vsr = sr / vel;

	assert(ttable.nrow() == locs1.nrow());
	assert(ttable.ncol() == locs2.nrow());

	#pragma omp parallel for
	for (size_t i = 0; i < ttable.nrow(); ++i)
	{
		float *sloc = locs1.row(i);
		uint16_t *tt_row = ttable.row(i);

		for (size_t j = 0; j < ttable.ncol(); ++j) 
		{
			float dist = DistCartesian(sloc, locs2.row(j));			
			tt_row[j] = static_cast<uint16_t>(dist * vsr + 0.5);
		}
	}
}



}

