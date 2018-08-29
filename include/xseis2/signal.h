#ifndef SIGNAL_H
#define SIGNAL_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "xseis2/globals.h"
#include "gsl/span"


namespace xseis {


template<typename Container>
float Max(Container& data) {
	return *std::max_element(data.begin(), data.end());
}

template<typename Container>
float Min(Container& data) {
	return *std::min_element(data.begin(), data.end());
}

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

uint mod_floor(int a, int n) {
	return ((a % n) + n) % n;
}


void BuildPhaseShiftVec(gsl::span<Complex> vec, int const nshift) {
	
	uint32_t nfreq = vec.size();
	float const fstep = 0.5 / (nfreq - 1);
	float const factor = nshift * 2 * M_PI * fstep;

	for(size_t i = 0; i < nfreq; ++i) {
		vec[i][0] = std::cos(i * factor);
		vec[i][1] = std::sin(i * factor);			
	}
}


// Mutiply sig1 by sig2 (x + yi)(u + vi) = (xu-yv) + (xv+yu)i
// x + yi = s1[0] + s1[1]i
// u + vi = s2[0] + s2[1]i
#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
void Convolve(Complex const* const sig2, Complex* const sig1, uint32_t const nfreq)
{
	float tmp;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		tmp = sig1[i][0] * sig2[i][0] - sig1[i][1] * sig2[i][1];
		sig1[i][1] = sig1[i][0] * sig2[i][1] + sig1[i][1] * sig2[i][0];
		sig1[i][0] = tmp;
	}
}

#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
inline void Convolve(Complex const* const sig1, Complex const* const sig2,
		   Complex* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i][0] = sig1[i][0] * sig2[i][0] - sig1[i][1] * sig2[i][1];
		out[i][1] = sig1[i][0] * sig2[i][1] + sig1[i][1] * sig2[i][0];
	}
}


#pragma omp declare simd aligned(data, stack:MEM_ALIGNMENT)
inline void Accumulate(Complex const* const data, Complex* const stack,
						 uint32_t const npts)
{		
	#pragma omp simd aligned(data, stack:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		stack[i][0] += data[i][0];
		stack[i][1] += data[i][1];
	}
}

#pragma omp declare simd aligned(data, stack:MEM_ALIGNMENT)
inline void Accumulate(float const* const data, float* const stack,
						 uint32_t const npts)
{		
	#pragma omp simd aligned(data, stack:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		stack[i] += data[i];
	}
}

#pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
void Whiten(Complex* const sig, uint32_t const npts)
{		
	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		float abs = std::sqrt(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1]);
		sig[i][0] /= abs;
		sig[i][1] /= abs;
	}
}

#pragma omp declare simd aligned(sig, out:MEM_ALIGNMENT)
void Absolute(Complex const* const sig, float* out, uint32_t const npts)
{		
	#pragma omp simd aligned(sig, out:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		out[i] = std::sqrt(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1]);
	}
}

#pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
void Absolute(float* sig, uint32_t const npts)
{		
	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		sig[i] = std::abs(sig[i]);
	}
}


// Cross-correlate complex signals, cc(f) = s1(f) x s2*(f)
#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
void XCorr(Complex const* const sig1, Complex const* const sig2,
		   Complex* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i][0] = sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1];
		out[i][1] = sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0];
	}
}

#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
float DotProductEnergy(float const* const sig1, float const* const sig2, uint32_t const npts)
{
	float result = 0;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < npts; ++i){
		// result += sig1[0] * sig2[0];		
		result += (sig1[0] * sig2[0]) * (sig1[0] * sig2[0]);		
	}
	return result;
}

#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
float DotProduct(float const* const sig1, float const* const sig2, uint32_t const npts)
{
	float result = 0;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < npts; ++i){
		result += sig1[0] * sig2[0];		
	}
	return result;
}


void TaperCosine(gsl::span<float> sig, uint32_t const len_taper)
{
	float const factor = (2 * M_PI) / ((len_taper * 2) - 1);

	for (size_t i = 0; i < len_taper; ++i) {
		sig[i] *= 0.5 - 0.5 * std::cos(i * factor);
	}

	for (size_t i = 0; i < len_taper; ++i) {
		sig[sig.size() - len_taper + i] *= 0.5 - 0.5 * std::cos((i + len_taper) * factor);
	}
}


void BuildFreqFilter(const std::vector<float>& corner_freqs, float const sr, gsl::span<float> filter)
{

	uint32_t nfreq = filter.size();
	float fsr = (nfreq * 2 - 1) / sr;
	// printf("nfreq: %u, FSR: %.4f\n", nfreq, fsr);

	std::vector<uint32_t> cx;
	for(auto&& cf : corner_freqs) {
		cx.push_back(static_cast<uint32_t>(cf * fsr + 0.5));
		// printf("cf/fsr %.2f, %.5f\n", cf, fsr);
	}
	// printf("filt corner indexes \n");
	// for(auto&& c : cx) {
	// 	// printf("cx/ cast: %.3f, %u\n", cx, (uint32_t)cx);
	// 	printf("--%u--", c);
	// }
	// printf("\n");

	// whiten corners:  cutmin--porte1---porte2--cutmax

	for(auto& x : filter) {x = 0;}

	// int wlen = porte1 - cutmin;
	float cosm_left = M_PI / (2. * (cx[1] - cx[0]));
	// left hand taper
	for (uint32_t i = cx[0]; i < cx[1]; ++i) {
		filter[i] = std::pow(std::cos((cx[1] - (i + 1) ) * cosm_left), 2.0);
	}

	// setin middle freqs amp = 1
	for (uint32_t i = cx[1]; i < cx[2]; ++i) {
		filter[i] = 1;
	}

	float cosm_right = M_PI / (2. * (cx[3] - cx[2]));

	// right hand taper
	for (uint32_t i = cx[2]; i < cx[3]; ++i) {
		filter[i] = std::pow(std::cos((i - cx[2]) * cosm_right), 2.0);
	}

}

void ApplyFreqFilterReplace(const gsl::span<float> filter, gsl::span<Complex> fsig)
{
	for (uint32_t i = 0; i < filter.size(); ++i)
	{
		if(filter[i] == 0) {
			fsig[i][0] = 0;
			fsig[i][1] = 0;
		}
		else {
			float angle = std::atan2(fsig[i][1], fsig[i][0]);
			fsig[i][0] = filter[i] * std::cos(angle);
			fsig[i][1] = filter[i] * std::sin(angle);
		}		
	}
}


void Multiply(float *sig, size_t npts, float val){
	for (size_t i = 0; i < npts; ++i){
		sig[i] *= val;
	}
}

void Multiply(Complex* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i][0] *= val;
		data[i][1] *= val;
	}
}


template<typename T>
void Multiply(gsl::span<T> data, float val) {
	Multiply(&data[0], data.size(), val);
}

void Fill(Complex* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i][0] = val;
		data[i][1] = val;
	}
}

void Fill(float* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i] = val;		
	}
}



}

#endif



