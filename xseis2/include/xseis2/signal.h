#pragma once

#include <omp.h>
#include <fftw3.h>
#include "xseis2/core.h"


namespace xs {

// const int FFTW_PATIENCE = FFTW_ESTIMATE;
// const int FFTW_PATIENCE = FFTW_MEASURE;
const int FFTW_PATIENCE = FFTW_PATIENT;
// const int FFTW_PATIENCE = FFTW_EXHAUSTIVE;
// const int FFTW_PATIENCE = FFTW_WISDOM_ONLY;


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


inline float DistCartesian(gsl::span<float> a, gsl::span<float> b)
{	
	return DistCartesian(a.data(), b.data());
}

inline float DistCartesian2D(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	return std::sqrt(dx * dx + dy * dy);
}


float DistDiff(float* a, float* b, float* c) {	
	return DistCartesian(a, c) - DistCartesian(b, c);
}


template<typename T>
void Roll(gsl::span<T> sig, long nroll)
{
	assert(nroll < sig.size());
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
int64_t ArgMax(gsl::span<T> data) {
	return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

template<typename T>
int64_t ArgMin(gsl::span<T> data) {
	return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
}

// uint32_t mod_floor(int a, int n) {
// 	return ((a % n) + n) % n;
// }


#pragma omp declare simd aligned(sig1, sig2:MIN_ALIGN)
void Convolve(Complex32 const* const sig2, Complex32* const sig1, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2:MIN_ALIGN)
	for (uint32_t i = 0; i < nfreq; ++i) sig1[i] *= sig2[i];
}


#pragma omp declare simd aligned(data, stack:MIN_ALIGN)
inline void Accumulate(float const* const data, float* const stack,
						 uint32_t const npts)
{		
	#pragma omp simd aligned(data, stack:MIN_ALIGN)
	for(uint32_t i = 0; i < npts; ++i) {
		stack[i] += data[i];
	}
}

#pragma omp declare simd aligned(sig:MIN_ALIGN)
void Whiten(Complex32* const sig, uint32_t const npts)
{		
	#pragma omp simd aligned(sig:MIN_ALIGN)
	for(uint32_t i = 0; i < npts; ++i) sig[i] /= std::abs(sig[i]);		
}


// #pragma omp declare simd aligned(sig:MIN_ALIGN)
// void Absolute(float* sig, uint32_t const npts)
// {		
// 	#pragma omp simd aligned(sig:MIN_ALIGN)
// 	for(uint32_t i = 0; i < npts; ++i) {
// 		sig[i] = std::abs(sig[i]);
// 	}
// }

void AbsCopy(gsl::span<Complex32> const in, gsl::span<float> out)
{	
	// #pragma omp simd aligned(sig:MIN_ALIGN)
	for(uint32_t i = 0; i < in.size(); ++i) {
		out[i] = std::abs(in[i]);
	}
}


template<typename T>
void Absolute(gsl::span<T> data) {
	for(auto& x : data) x = std::abs(x);
}



// void NormCopy(gsl::span<Complex32> const in, gsl::span<float> out)
// {	
// 	// #pragma omp simd aligned(sig:MIN_ALIGN)
// 	for(uint32_t i = 0; i < in.size(); ++i) {
// 		out[i] = std::norm(in[i]);
// 	}
// }


#pragma omp declare simd aligned(sig1, sig2, out:MIN_ALIGN)
void XCorr(Complex32 const* const sig1, Complex32 const* const sig2,
		   Complex32* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MIN_ALIGN)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i] = std::conj(sig1[i]) * sig2[i];
	}
}

// #pragma omp declare simd aligned(sig1, sig2:MIN_ALIGN)
// float DotProductEnergy(float const* const sig1, float const* const sig2, uint32_t const npts)
// {
// 	float result = 0;
// 	#pragma omp simd aligned(sig1, sig2:MIN_ALIGN)
// 	for (uint32_t i = 0; i < npts; ++i){
// 		// result += sig1[0] * sig2[0];		
// 		result += (sig1[0] * sig2[0]) * (sig1[0] * sig2[0]);		
// 	}
// 	return result;
// }

// #pragma omp declare simd aligned(sig1, sig2:MIN_ALIGN)
// float DotProduct(float const* const sig1, float const* const sig2, uint32_t const npts)
// {
// 	float result = 0;
// 	#pragma omp simd aligned(sig1, sig2:MIN_ALIGN)
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
void Multiply(gsl::span<T> data, float const val) {
	// Multiply(data.data(), data.size(), val);
	for(auto& x : data) x *= val;
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
	std::copy(in.data(), in.data() + in.size(), out.data());
}

std::vector<float> RowMaxes (const VecOfSpans<float> dat)
{	
	std::vector<float> maxes;
	maxes.reserve(dat.size());
	for(auto&& x : dat) maxes.emplace_back(Max(x));
	return maxes;	
}

template<typename T>
T Mean (const gsl::span<T> dat)
{	
	T sum = 0;
	for(auto&& x : dat) sum += x;
	return sum / static_cast<T>(dat.size());		
}

// assume in is absolute valued
void SlidingWinMax(const gsl::span<float> in, gsl::span<float> out, uint32_t wlen)
{	
	assert(wlen >= 3);
	assert(in.size() == out.size());
	uint32_t npts = in.size();

	if (wlen % 2 == 0) wlen += 1;
	uint32_t hlen = wlen / 2 + 1;

	for(uint32_t i = 0; i < hlen; ++i) {
		out[i] = *std::max_element(&in[0], &in[hlen]);
	}

	for(uint32_t i = npts - hlen; i < npts; ++i) {
		out[i] = *std::max_element(&in[npts - hlen], &in[npts - 1]);
	}

	// handle non-edge case
	for (uint32_t i = hlen; i < npts - hlen; ++i) {
		out[i] = *std::max_element(&in[i - hlen], &in[i + hlen]);
	}

}


Vector<Complex32> BuildPhaseShiftVec(size_t const nfreq, int const nshift) {

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


Array2D<float> ZeroPad(Array2D<float>& dat, size_t npad) 
{
	auto out = Array2D<float>(dat.nrow(), npad);

	for(size_t i = 0; i < dat.nrow(); ++i) {
		Fill(out.span(i), 0.0f);
		Copy(dat.row(i), dat.ncol(), out.row(i));
	}

	return out;
}

Vector<float> SumRows(VecOfSpans<float> dat) 
{
	size_t npts = dat[0].size();
	auto out = Vector<float>(npts);
	Fill(out.span(), 0.0f);

	for(size_t i = 0; i < dat.size(); ++i) {
		Accumulate(dat[i].data(), out.data(), npts);
	}
	return out;
}


// Forward fft of data and whitens between corner freqs, data is modified
Array2D<Complex32> FFTAndWhiten(Array2D<float>& dat, float const sr, std::vector<float> cfreqs, float const taper_len=0.02) 
{

	size_t const nchan = dat.nrow();
	size_t const wlen = dat.ncol();
	uint32_t const taper_nsamp = taper_len * wlen;
	size_t const nfreq = wlen / 2 + 1;

	auto fdat = Array2D<Complex32>(nchan, nfreq);
	auto filter = BuildFreqFilter(cfreqs, nfreq, sr);

	auto buf = Vector<float>(wlen);
	auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(0));	
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, buf.data(), fptr, FFTW_PATIENCE);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fptr, buf.data(), FFTW_PATIENCE);

	#pragma omp parallel for
	for(size_t i = 0; i < dat.nrow(); ++i) {
		TaperCosine(dat.span(i), taper_nsamp);
		auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(i));
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fptr); // fwd fft
		ApplyFreqFilterReplace(filter.span(), fdat.span(i)); // apply whiten
		fftwf_execute_dft_c2r(plan_inv, fptr, dat.row(i)); // inv fft
		TaperCosine(dat.span(i), taper_nsamp); 
		Multiply(dat.span(i), 1.0f / static_cast<float>(wlen)); // fix fftw3 scaling
		fftwf_execute_dft_r2c(plan_fwd, dat.row(i), fptr);
	}

	return fdat;
}


// ccf for each sta pair = stacked envelopes of inter-channel ccfs
// set smoothin window length to account for uncertainties in vel model
void XCorrCombineChans(Array2D<Complex32>& fdat, KeyGroups& groups, VecOfSpans<uint16_t> pairs, Array2D<float>& ccdat, uint32_t wlen_smooth=0) 
{
	assert((uintptr_t) ccdat.row(1) % MIN_ALIGN == 0);
	assert((uintptr_t) fdat.row(1) % MIN_ALIGN == 0);

	uint32_t wlen = ccdat.ncol();
	uint32_t nfreq = fdat.ncol();

	// convolve complex sig with vshift for zero lag at middle sample
	Vector<Complex32> vshift = BuildPhaseShiftVec(nfreq, wlen / 2);
	// assumes whitened signals (i.e energy same for all)
	float energy = Energy(fdat.span(0)) * 2;

	auto fb = Vector<Complex32>(nfreq); // only used for planning to not destroy fdat
	auto fptr = reinterpret_cast<fftwf_complex*>(fb.data());	
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fptr, ccdat.row(0), FFTW_PATIENCE);

	#pragma omp parallel
	{
		auto buf = Vector<float>(wlen);
		auto fbuf = Vector<Complex32>(nfreq);
		auto fptr = reinterpret_cast<fftwf_complex*>(fbuf.data());

		#pragma omp for
		for(size_t i = 0; i < pairs.size(); ++i) {

			auto& pair = pairs[i];
			auto csig = ccdat.span(i);
			Fill(csig, 0.0f);

			uint32_t nstack = 0;

			for(auto&& k0 : groups[pair[0]]) {
				for(auto&& k1 : groups[pair[1]]) {
					XCorr(fdat.row(k0), fdat.row(k1), fbuf.data(), fbuf.size());
					Convolve(vshift.data(), fbuf.data(), fbuf.size());
					fftwf_execute_dft_c2r(plan_inv, fptr, buf.data());
					Multiply(buf.span(), 1.0f / energy);
					for(size_t j=0; j < buf.size(); ++j) {
						csig[j] += std::abs(buf[j]);
					}		
					nstack++;					
				}
			}
			Multiply(csig, 1.0f / static_cast<float>(nstack));
			Copy(csig, buf.span());
			if(wlen_smooth != 0) SlidingWinMax(buf.span(), csig, wlen_smooth);
		}
	}
}


// ccf for each sta pair = stacked envelopes of inter-channel ccfs
// set smoothin window length to account for uncertainties in vel model
void XCorrChans(Array2D<Complex32>& fdat, VecOfSpans<uint16_t> pairs, Array2D<float>& ccdat, uint32_t wlen_smooth=0) 
{
	assert((uintptr_t) ccdat.row(1) % MIN_ALIGN == 0);
	assert((uintptr_t) fdat.row(1) % MIN_ALIGN == 0);

	uint32_t wlen = ccdat.ncol();
	uint32_t nfreq = fdat.ncol();

	// convolve complex sig with vshift for zero lag at middle sample
	Vector<Complex32> vshift = BuildPhaseShiftVec(nfreq, wlen / 2);
	// assumes whitened signals (i.e energy same for all)
	float energy = Energy(fdat.span(0)) * 2;

	auto fb = Vector<Complex32>(nfreq); // only used for planning to not destroy fdat
	auto fptr = reinterpret_cast<fftwf_complex*>(fb.data());	
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fptr, ccdat.row(0), FFTW_PATIENCE);

	#pragma omp parallel
	{
		auto buf = Vector<float>(wlen);
		auto fbuf = Vector<Complex32>(nfreq);
		auto fptr = reinterpret_cast<fftwf_complex*>(fbuf.data());

		#pragma omp for
		for(size_t i = 0; i < pairs.size(); ++i) {

			auto k0 = pairs[i][0];
			auto k1 = pairs[i][1];
			
			XCorr(fdat.row(k0), fdat.row(k1), fbuf.data(), fbuf.size());
			Convolve(vshift.data(), fbuf.data(), fbuf.size());
			fftwf_execute_dft_c2r(plan_inv, fptr, buf.data());
			Multiply(buf.span(), 1.0f / energy);

			auto csig = ccdat.span(i);
			// Fill(csig, 0.0f);
			// Copy(csig, buf.span());
			if(wlen_smooth == 0)
			{
				Copy(buf.span(), csig);				
			}
			else
			{
				Absolute(buf.span());
				SlidingWinMax(buf.span(), csig, wlen_smooth);
			}
		}
	}
}


// Stack buffer is twice length of signal to prevent rollover
// argmax_true = argmax(stack) - len(stack) / 2
Vector<float> RollAndStack(VecOfSpans<float> const signals, const KeyGroups& groups, gsl::span<uint16_t> const shifts) 
{	
	uint32_t npts = signals[0].size();
	auto stack = Vector<float>(npts * 2);
	Fill(stack.span(), 0.0f);

	for(size_t i = 0; i < groups.size(); ++i)
	{
		int rollby = static_cast<int>(shifts[i]);

		for(auto& key : groups[i])
		{
			float* out = &stack[npts] - rollby;
			Accumulate(signals[key].data(), out, signals[key].size());
		}
	}

	Multiply(stack.span(), 1.0f / signals.size());
	return stack;
}

void RollSigs(VecOfSpans<float> signals, const KeyGroups& groups, gsl::span<uint16_t> const shifts) 
{	

	for(size_t i = 0; i < groups.size(); ++i) {
		int rollby = static_cast<int>(shifts[i]);
		for(auto& key : groups[i]) Roll(signals[key], rollby);			
	}
}

void RollSigsHack(VecOfSpans<float> signals, const KeyGroups& groups, gsl::span<uint16_t> const shifts) 
{	
	auto min = Min(shifts);
	for(auto& x : shifts) x -= min;
		
	uint32_t nsamp = signals[0].size();

	for(size_t i = 0; i < groups.size(); ++i) {
		int rollby = static_cast<int>(shifts[i]);
		if (rollby < nsamp) for(auto& key : groups[i]) Roll(signals[key], rollby);
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
	// uint32_t nfreq_r2c = wlen / 2 + 1;

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


float MADMax(gsl::span<float> power, uint32_t scale=1) {
	// destroys input
	size_t npts = power.size();
	auto ptr = power.data();

	uint32_t amax = ArgMax(power);
	float vmax = power[amax];

	size_t half = npts / 2;
	std::nth_element(ptr, ptr + half, ptr + npts);
	float med = ptr[half];
	// float mean = process::mean(ptr, npts);

	for(size_t i = 0; i < npts; ++i) {
		ptr[i] = std::abs(ptr[i] - med);
	}

	std::nth_element(ptr, ptr + half, ptr + npts);
	float mad = ptr[half];
	// float mdev = (vmax - med) / mad;
	// uint32_t mdev = static_cast<uint32_t>((vmax - med) / mad * scale) ;
	float mdev = (vmax - med) / mad * scale;	
	return mdev;
}


// void FFTWImportWisdom()
// {
 
// std::string wisdom = " ";
// // fftwf_import_wisdom_from_string
// }



// Array2D<Complex32> WhitenAndFFTPadDec2x(Array2D<float>& dat, float& sr, std::vector<float> cfreqs, float taper_len=0.02) 
// {

// 	size_t nchan = dat.nrow();
// 	size_t wlen = dat.ncol();
// 	size_t nfreq = wlen / 2 + 1;
// 	uint32_t taper_nsamp = taper_len * wlen;

// 	// size_t wlen_pad = wlen * 2;
// 	// size_t nfreq_pad = wlen_pad / 2 + 1;

// 	auto fdat = Array2D<Complex32>(nchan, nfreq);
// 	auto filter = BuildFreqFilter(cfreqs, nfreq, sr);
// 	// float energy = Energy(filter.span()) * 2;

// 	auto buf = Vector<float>(wlen);

// 	// auto fbuf = Vector<Complex32>(wlen);
// 	auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(0));	
// 	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, buf.data(), fptr, FFTW_PATIENCE);
// 	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, fptr, buf.data(), FFTW_PATIENCE);
	
// 	#pragma omp parallel for
// 	for(size_t i = 0; i < dat.nrow(); ++i) {
// 		auto fptr = reinterpret_cast<fftwf_complex*>(fdat.row(i));
// 		auto sig = dat.span(i);
// 		fftwf_execute_dft_r2c(plan_fwd, sig.data(), fptr);
// 		ApplyFreqFilterReplace(filter.span(), fdat.span(i)); // apply whiten
// 		fftwf_execute_dft_c2r(plan_inv, fptr, sig.data());
// 		TaperCosine(sig, taper_nsamp);
// 		Multiply(sig, 1.0f / static_cast<float>(wlen));		
// 		for(size_t j = 0; j < wlen / 2; ++j) sig[j] = sig[j * 2];
// 		for(size_t j = wlen / 2; j < wlen; ++j) sig[j] = 0;

// 		fftwf_execute_dft_r2c(plan_fwd, sig.data(), fptr);
// 	}
// 	sr /= 2;

// 	return fdat;
// }


// // ccf for each sta pair = stacked envelopes of inter-channel ccfs
// void XCorrChanGroupsEnvelope(Array2D<Complex32>& fdat, KeyGroups& groups, VecOfSpans<uint16_t> pairs, Array2D<float>& ccdat) 
// {
// 	uint32_t wlen = ccdat.ncol();
// 	uint32_t nfreq = fdat.ncol();

// 	// values to roll ccfs for zero lag in middle (conv in freq domain)
// 	auto vshift = BuildPhaseShiftVec(nfreq, wlen / 2);
// 	float energy = Energy(fdat.span(0)) * 2;
// 	std::cout << "energy: " << energy << "\n";


// 	auto fb = Vector<Complex32>(wlen); // only used for planning to not destroy fdat
// 	auto fptr = reinterpret_cast<fftwf_complex*>(fb.data());	
// 	fftwf_plan plan_c2c = fftwf_plan_dft_1d(wlen, fptr, fptr, FFTW_BACKWARD, FFTW_PATIENCE);

// 	#pragma omp parallel
// 	{
// 		auto fbuf = Vector<Complex32>(wlen);
// 		auto fptr = reinterpret_cast<fftwf_complex*>(fbuf.data());

// 		auto fbuf_pos = gsl::make_span(fbuf.data(), fbuf.data() + wlen / 2);
// 		auto fbuf_neg = gsl::make_span(fbuf.data() + wlen / 2, fbuf.end());

// 		#pragma omp for
// 		for(size_t i = 0; i < pairs.size(); ++i) {
// 			auto pair = pairs[i];
// 			float *csig = ccdat.row(i);
// 			std::fill(csig, csig + wlen, 0);

// 			uint32_t nstack = 0;		
// 			for(auto&& k0 : groups[pair[0]]) {
// 				for(auto&& k1 : groups[pair[1]]) {
// 					Fill(fbuf_neg, {0.0f, 0.0f});
// 					XCorr(fdat.row(k0), fdat.row(k1), fbuf_pos.data(), fbuf_pos.size());
// 					// Multiply(fbuf_pos.subspan(1), {2.0f, 0.0f});
// 					Multiply(fbuf_pos.subspan(1), 2.0f);
// 					Convolve(&vshift[0], fbuf_pos.data(), fbuf_pos.size());

// 					fftwf_execute_dft(plan_c2c, fptr, fptr);
// 					Multiply(fbuf.span(), 1.0f / energy);
// 					// Multiply(fbuf.span(), {1.0f / energy, 0.0f});
// 					// Absolute(fbuf.data(), wlen, ccdat.row(nstack));
// 					// AbsCopy(fbuf.span(), dat.span(k));

// 					for(size_t j=0; j < fbuf.size(); ++j)
// 					{
// 						csig[j] += std::norm(fbuf[j]);	
// 					} 
// 					nstack++;
// 				}
// 			}
// 			// Multiply(csig, wlen, 1.0f / static_cast<float>(nstack));		
// 			Multiply(gsl::make_span(csig, wlen), 1.0f / static_cast<float>(nstack));		
// 		}
// 	}
// }



}




