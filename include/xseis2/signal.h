#pragma once

#include <omp.h>
#include <fftw3.h>
#include "xseis2/core.h"


namespace xseis {

// const int FFTW_PATIENCE = FFTW_ESTIMATE;
// const int FFTW_PATIENCE = FFTW_MEASURE;
const int FFTW_PATIENCE = FFTW_PATIENT;
// const int FFTW_PATIENCE = FFTW_WISDOM_ONLY;

template<typename T>
void Roll(gsl::span<T> sig, long nroll)
{
	std::rotate(sig.begin(), sig.begin() + nroll, sig.end());
}

template<typename T>
T Max(gsl::span<T> data) {
	return *std::max_element(data.begin(), data.end());
}

template<typename T>
T Min(gsl::span<T> data) {
	return *std::min_element(data.begin(), data.end());
}

template<typename T>
size_t ArgMax(gsl::span<T> data) {
	return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

template<typename T>
size_t ArgMin(gsl::span<T> data) {
	return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
}

// uint32_t mod_floor(int a, int n) {
// 	return ((a % n) + n) % n;
// }


#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
void Convolve(Complex32 const* const sig2, Complex32* const sig1, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i) sig1[i] *= sig2[i];
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

// #pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
// void Absolute(float* sig, uint32_t const npts)
// {		
// 	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
// 	for(uint32_t i = 0; i < npts; ++i) {
// 		sig[i] = std::abs(sig[i]);
// 	}
// }

void AbsCopy(gsl::span<Complex32> const in, gsl::span<float> out)
{	
	// #pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < in.size(); ++i) {
		out[i] = std::abs(in[i]);
	}
}


// void NormCopy(gsl::span<Complex32> const in, gsl::span<float> out)
// {	
// 	// #pragma omp simd aligned(sig:MEM_ALIGNMENT)
// 	for(uint32_t i = 0; i < in.size(); ++i) {
// 		out[i] = std::norm(in[i]);
// 	}
// }


#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
void XCorr(Complex32 const* const sig1, Complex32 const* const sig2,
		   Complex32* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i] = std::conj(sig1[i]) * sig2[i];
	}
}

// #pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
// float DotProductEnergy(float const* const sig1, float const* const sig2, uint32_t const npts)
// {
// 	float result = 0;
// 	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
// 	for (uint32_t i = 0; i < npts; ++i){
// 		// result += sig1[0] * sig2[0];		
// 		result += (sig1[0] * sig2[0]) * (sig1[0] * sig2[0]);		
// 	}
// 	return result;
// }

// #pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
// float DotProduct(float const* const sig1, float const* const sig2, uint32_t const npts)
// {
// 	float result = 0;
// 	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
// 	for (uint32_t i = 0; i < npts; ++i){
// 		result += sig1[0] * sig2[0];		
// 	}
// 	return result;
// }


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


// void Multiply(float *sig, size_t npts, float val){
// 	for (size_t i = 0; i < npts; ++i){
// 		sig[i] *= val;
// 	}
// }

// void Multiply(Complex* data, size_t npts, float val)
// {		
// 	for(size_t i = 0; i < npts; ++i) {
// 		data[i][0] *= val;
// 		data[i][1] *= val;
// 	}
// }


template<typename T>
void Multiply(gsl::span<T> data, float val) {
	// Multiply(data.data(), data.size(), val);
	for(auto&& x : data) x *= val;
}

// void Fill(Complex* data, size_t npts, float val)
// {		
// 	for(size_t i = 0; i < npts; ++i) {
// 		data[i][0] = val;
// 		data[i][1] = val;
// 	}
// }

// void Fill(float* data, size_t npts, float val)
// {		
// 	for(size_t i = 0; i < npts; ++i) {
// 		data[i] = val;		
// 	}
// }

template<typename T>
void Fill(gsl::span<T> data, T val) {
	// Fill(data.data(), data.size(), val);
	for(auto& x : data) x = val;
}

template<typename T>
void Stack(gsl::span<T> const in, gsl::span<T> stack) {
	for(size_t i = 0; i < in.size(); ++i) {stack[i] += in[i];}
}

float Energy(gsl::span<float> const data)
{
	float total = 0;
	for(auto&& x : data) total += x * x;	
	return total;
}

float Energy(gsl::span<Complex32> const data)
{
	float total = 0;
	for(auto&& x : data) total += std::norm(x);	
	return total;
}

template<typename T>
void Copy(T const *in, size_t npts, T *out)
{		
	std::copy(in, in + npts, out);
}

template<typename T>
void Copy(gsl::span<T> const in, gsl::span<T> out) {
	assert(in.size() == out.size());
	std::copy(in.data(), in.end(), out.data());
}

Vector<Complex32> BuildPhaseShiftVec(size_t nfreq, int const nshift) {

	auto vec = Vector<Complex32>(nfreq);
	float const fstep = 0.5f / (nfreq - 1.0f);
	float const factor = nshift * 2 * M_PI * fstep;

	for(size_t i = 0; i < nfreq; ++i) {
		vec[i] = {std::cos(i * factor), std::sin(i * factor)};
	}
	return vec;
}


Vector<float> BuildFreqFilter(std::vector<float>& corner_freqs, uint32_t nfreq, float sr)
{
	float fsr = (nfreq * 2 - 1) / sr;

	// whiten corners:  cutmin--porte1---porte2--cutmax
	std::vector<uint32_t> cx;
	for(auto&& cf : corner_freqs) cx.push_back(static_cast<uint32_t>(cf * fsr + 0.5f));

	auto filter = Vector<float>(nfreq);	
	Fill(filter.span(), 0.0f);	

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

void ApplyFreqFilterReplace(const gsl::span<float> filter, gsl::span<Complex32> fsig)
{
	for (uint32_t i = 0; i < filter.size(); ++i)
	{
		if(filter[i] == 0) {
			fsig[i] = {0, 0};
		}
		else {
			float angle = std::arg(fsig[i]);
			fsig[i] = {filter[i] * std::cos(angle), filter[i] * std::sin(angle)};
		}		
	}
}


Array2D<Complex32> WhitenAndFFT(Array2D<float>& dat, float sr, std::vector<float> cfreqs) 
{
	size_t nchan = dat.nrow();
	size_t wlen = dat.ncol();
	size_t nfreq = wlen / 2 + 1;
	uint32_t taper_nsamp = 100;

	auto fdat = Array2D<Complex32>(nchan, nfreq);

	auto filter = BuildFreqFilter(cfreqs, nfreq, sr);
	// float energy = Energy(filter.span()) * 2;

	auto buf = Vector<float>(wlen);
	auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(0));	
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, buf.data(), fptr, FFTW_PATIENCE);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fptr, buf.data(), FFTW_PATIENCE);

	#pragma omp parallel for
	for(size_t i = 0; i < dat.nrow(); ++i) {
		auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(i));
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fptr);
		ApplyFreqFilterReplace(filter.span(), fdat.span(i)); // apply whiten
		fftwf_execute_dft_c2r(plan_inv, fptr, dat.row(i));
		TaperCosine(dat.span(i), taper_nsamp);
		Multiply(dat.span(i), 1.0f / static_cast<float>(wlen));
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fptr);
	}

	return fdat;
}


// ccf for each sta pair = stacked envelopes of inter-channel ccfs
void XCorrChanGroupsEnvelope(Array2D<Complex32>& fdat, KeyGroups& groups, VecOfSpans<uint16_t> pairs, Array2D<float>& ccdat) 
{
	uint32_t wlen = ccdat.ncol();
	uint32_t nfreq = fdat.ncol();

	// values to roll ccfs for zero lag in middle (conv in freq domain)
	auto vshift = BuildPhaseShiftVec(nfreq, wlen / 2);
	float energy = Energy(fdat.span(0)) * 2;
	std::cout << "energy: " << energy << "\n";


	auto fb = Vector<Complex32>(wlen); // only used for planning to not destroy fdat
	auto fptr = reinterpret_cast<fftwf_complex*>(fb.data());	
	fftwf_plan plan_c2c = fftwf_plan_dft_1d(wlen, fptr, fptr, FFTW_BACKWARD, FFTW_PATIENCE);

	#pragma omp parallel
	{
		auto fbuf = Vector<Complex32>(wlen);
		auto fptr = reinterpret_cast<fftwf_complex*>(fbuf.data());

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
					Fill(fbuf_neg, {0.0f, 0.0f});
					XCorr(fdat.row(k0), fdat.row(k1), fbuf_pos.data(), fbuf_pos.size());
					// Multiply(fbuf_pos.subspan(1), {2.0f, 0.0f});
					Multiply(fbuf_pos.subspan(1), 2.0f);
					Convolve(&vshift[0], fbuf_pos.data(), fbuf_pos.size());

					fftwf_execute_dft(plan_c2c, fptr, fptr);
					Multiply(fbuf.span(), 1.0f / energy);
					// Multiply(fbuf.span(), {1.0f / energy, 0.0f});

					// Absolute(fbuf.data(), wlen, ccdat.row(nstack));
					// AbsCopy(fbuf.span(), dat.span(k));

					for(size_t j=0; j < fbuf.size(); ++j)
					{
						// csig[j] += fbuf[j][0] * fbuf[j][0] + fbuf[j][1] * fbuf[j][1];	
						csig[j] += std::norm(fbuf[j]);	
					} 
					nstack++;
				}
			}
			// Multiply(csig, wlen, 1.0f / static_cast<float>(nstack));		
			Multiply(gsl::make_span(csig, wlen), 1.0f / static_cast<float>(nstack));		
		}
	}
}


void RollSigs(VecOfSpans<float> signals, const KeyGroups& groups, gsl::span<uint16_t> const shifts) 
{	
	for(size_t i = 0; i < groups.size(); ++i) {
		int rollby = static_cast<int>(shifts[i]);
		for(auto& key : groups[i]) Roll(signals[key], rollby);			
	}
}

Vector<float> StackSigs(VecOfSpans<float> const signals) 
{	
	auto stack = Vector<float>(signals[0].size());
	Fill(stack.span(), 0.0f);
	for(auto&& sig : signals) Stack(sig, stack.span());
	Multiply(stack.span(), 1.0f / signals.size());
	return stack;
}


void Envelope(VecOfSpans<float> signals) 
{
	uint32_t wlen = signals[0].size();
	uint32_t nfreq_r2c = wlen / 2 + 1;

	auto buf = Vector<float>(wlen);
	auto fbuf = Vector<Complex32>(wlen);
	auto fptr = reinterpret_cast<fftwf_complex*>(fbuf.data());		

	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, buf.data(), fptr, FFTW_PATIENCE);
	fftwf_plan plan_c2c = fftwf_plan_dft_1d(wlen, fptr, fptr, FFTW_BACKWARD, FFTW_PATIENCE);
	auto fbuf_pos = gsl::make_span(fbuf.data(), fbuf.data() + wlen / 2);
	auto fbuf_neg = gsl::make_span(fbuf.data() + wlen / 2, fbuf.end());

	for(auto& sig : signals) {
		Fill(fbuf_neg, {0.0f, 0.0f});
		fftwf_execute_dft_r2c(plan_fwd, sig.data(), fptr);
		Multiply(fbuf_pos.subspan(1), 2.0f);
		fftwf_execute(plan_c2c);
		Multiply(fbuf.span(), 1.0 / static_cast<float>(wlen));		
		AbsCopy(fbuf.span(), sig);		
	}
}

}




