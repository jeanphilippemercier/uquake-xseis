#ifndef SIGNAL_H
#define SIGNAL_H

#include <omp.h>
#include <fftw3.h>
#include "xseis2/core.h"


namespace xseis {

// const int FFTW_PATIENCE = FFTW_ESTIMATE;
// const int FFTW_PATIENCE = FFTW_MEASURE;
const int FFTW_PATIENCE = FFTW_PATIENT;
// const int FFTW_PATIENCE = FFTW_WISDOM_ONLY;


template<typename T>
T Max(gsl::span<T> data) {
	return *std::max_element(data.begin(), data.end());
}

template<typename T>
T Min(gsl::span<T> data) {
	return *std::min_element(data.begin(), data.end());
}


// template<typename Container>
// float Max(Container& data) {
// 	return *std::max_element(data.begin(), data.end());
// }

// template<typename Container>
// float Min(Container& data) {
// 	return *std::min_element(data.begin(), data.end());
// }

uint mod_floor(int a, int n) {
	return ((a % n) + n) % n;
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
void Absolute(Complex const* const sig, uint32_t const npts, float* out)
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

template<typename T>
void Fill(gsl::span<T> data, float val) {
	Fill(&data[0], data.size(), val);
}


float Energy(gsl::span<float> const data)
{
	float total = 0;
	for(auto&& i : data) total += i * i;	
	return total;
}

float Energy(gsl::span<Complex> const data)
{
	float total = 0;
	for(auto&& v : data) total += v[0] * v[0] + v[1] * v[1];	
	return total;
}	

void Copy(Complex const *in, size_t npts, Complex *out)
{		
	std::copy(&(in)[0][0], &(in + npts)[0][0], &out[0][0]);
}

void Copy(float const *in, size_t npts, float *out)
{		
	std::copy(in, in + npts, out);
}

// template<typename T>
// void Copy(gsl::span<T> data, float val) {
// 	Copy(&data[0], data.size(), val);
// }


Vector<Complex> BuildPhaseShiftVec(size_t nfreq, int const nshift) {

	auto vec = Vector<Complex>(nfreq);
	float const fstep = 0.5 / (nfreq - 1);
	float const factor = nshift * 2 * M_PI * fstep;

	for(size_t i = 0; i < nfreq; ++i) {
		vec[i][0] = std::cos(i * factor);
		vec[i][1] = std::sin(i * factor);			
	}
	return vec;
}


// void BuildFreqFilter(const std::vector<float>& corner_freqs, float const sr, gsl::span<float> filter)
Vector<float> BuildFreqFilter(std::vector<float>& corner_freqs, uint32_t nfreq, float sr)
{
	float fsr = (nfreq * 2 - 1) / sr;

	// whiten corners:  cutmin--porte1---porte2--cutmax
	std::vector<uint32_t> cx;
	for(auto&& cf : corner_freqs) cx.push_back(static_cast<uint32_t>(cf * fsr + 0.5));

	auto filter = Vector<float>(nfreq);	
	Fill(filter.span(), 0);	

	// int wlen = porte1 - cutmin;
	float cosm_left = M_PI / (2. * (cx[1] - cx[0]));
	// left hand taper
	for (uint32_t i = cx[0]; i < cx[1]; ++i) {
		float tmp = std::cos((cx[1] - (i + 1) ) * cosm_left);
		filter[i] = tmp * tmp;
	}

	// setin middle freqs amp = 1
	for (uint32_t i = cx[1]; i < cx[2]; ++i) filter[i] = 1;

	float cosm_right = M_PI / (2. * (cx[3] - cx[2]));

	// right hand taper
	for (uint32_t i = cx[2]; i < cx[3]; ++i) {
		float tmp = std::cos((i - cx[2]) * cosm_right);
		filter[i] = tmp * tmp;
	}

	return filter;
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



Array2D<Complex> WhitenAndFFT(Array2D<float>& dat, float sr, std::vector<float> cfreqs) 
{
	size_t nchan = dat.nrow();
	size_t wlen = dat.ncol();
	size_t nfreq = wlen / 2 + 1;
	uint32_t taper_nsamp = 100;

	auto fdat = Array2D<xseis::Complex>(nchan, nfreq);

	auto filter = BuildFreqFilter(cfreqs, nfreq, sr);
	// float energy = Energy(filter.span()) * 2;

	auto buf = Vector<float>(wlen);
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, buf.data(), fdat.row(0), FFTW_PATIENCE);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fdat.row(0), buf.data(), FFTW_PATIENCE);

	#pragma omp parallel for
	for(size_t i = 0; i < dat.nrow(); ++i) {
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fdat.row(i));
		ApplyFreqFilterReplace(filter.span(), fdat.span(i)); // apply whiten
		fftwf_execute_dft_c2r(plan_inv, fdat.row(i), dat.row(i));
		TaperCosine(dat.span(i), taper_nsamp);
		Multiply(dat.span(i), 1.0 / static_cast<float>(wlen));
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fdat.row(i));
	}

	return fdat;
}

// ccf for each sta pair = stacked envelopes of inter-channel ccfs
void XCorrChanGroupsEnvelope(Array2D<Complex>& fdat, KeyGroups& groups, VecOfSpans<uint16_t> pairs, Array2D<float>& ccdat) 
{
	uint32_t wlen = ccdat.ncol();
	uint32_t nfreq = fdat.ncol();

	// values to roll ccfs for zero lag in middle (conv in freq domain)
	auto vshift = xseis::BuildPhaseShiftVec(nfreq, wlen / 2);
	float energy = xseis::Energy(fdat.span(0)) * 2;
	std::cout << "energy: " << energy << "\n";

	// auto ftmp = xseis::Vector<xseis::Complex>(fdat.ncol());
	// fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(ccdat.ncol(), ftmp.data(), ccdat.data(), FFTW_PATIENCE);


	auto fb = xseis::Vector<xseis::Complex>(wlen); // only used for planning to not destroy fdat
	fftwf_plan plan_c2c = fftwf_plan_dft_1d(wlen, fb.data(), fb.data(), FFTW_BACKWARD, FFTW_PATIENCE);

	#pragma omp parallel
	{
		auto fbuf = xseis::Vector<xseis::Complex>(wlen);
		auto fbuf_pos = gsl::make_span(fbuf.data(), fbuf.data() + wlen / 2);
		auto fbuf_neg = gsl::make_span(fbuf.data() + wlen / 2, fbuf.end());

		#pragma omp for
		for(size_t i = 0; i < pairs.size(); ++i) {
			auto pair = pairs[i];
			float *csig = ccdat.row(i);
			std::fill(csig, csig + wlen, 0);

			uint32_t nstack = 0;		
			for(auto&& k0 : groups[pair[0]]) {
				for(auto&& k1 : groups[pair[1]]) {
					xseis::Fill(fbuf_neg, 0);
					xseis::XCorr(fdat.row(k0), fdat.row(k1), fbuf_pos.data(), fbuf_pos.size());
					xseis::Multiply(fbuf_pos.subspan(1), 2.0);
					xseis::Convolve(&vshift[0], fbuf_pos.data(), fbuf_pos.size());

					fftwf_execute_dft(plan_c2c, fbuf.data(), fbuf.data());
					xseis::Multiply(fbuf.span(), 1.0 / energy);

					// xseis::Absolute(fbuf.data(), wlen, ccdat.row(nstack));
					for(size_t j=0; j < fbuf.size(); ++j)
					{
						csig[j] += fbuf[j][0] * fbuf[j][0] + fbuf[j][1] * fbuf[j][1];	
					} 
					nstack++;
				}
			}
			xseis::Multiply(csig, wlen, 1.0 / nstack);		
		}

	}

}


	// // compute abs-valued cross-correlations (1 ccf per valid station pair)
	// #pragma omp parallel
	// {
	// 	auto tmp = malloc_cache_align<float>(wl);
	// 	auto ftmp = malloc_cache_align<fftwf_complex>(fl);

	// 	#pragma omp for
	// 	for(size_t i = 0; i < npair; ++i) {
	// 		uint16_t *pair = spairs.row(i);
	// 		float *csig = ccdat.row(i);
	// 		std::fill(csig, csig + wl, 0);

	// 		// sums absolute valued ccfs of all interstation channel pairs
	// 		uint32_t nstack = 0;		
	// 		for(auto&& k0 : groups[pair[0]]) {
	// 			for(auto&& k1 : groups[pair[1]]) {

	// 				process::XCorr(fdat.row(k0), fdat.row(k1), ftmp, fl);
	// 				process::Convolve(&vshift[0], ftmp, fl);
	// 				fftwf_execute_dft_c2r(plan_inv, ftmp, tmp);
	// 				for(size_t j=0; j < wl; ++j) csig[j] += std::abs(tmp[j]);
	// 				nstack++;
	// 			}
	// 		}
	// 		process::Multiply(csig, wl, 1.0 / (nstack * energy)); // normalize
	// 		process::EMA_NoAbs(csig, wl, cc_smooth_len, true); // EMA smoothing
	// 	}
	// 	free(tmp);
	// 	free(ftmp);
	// }
	// clock.log("xcorr");




}

#endif



