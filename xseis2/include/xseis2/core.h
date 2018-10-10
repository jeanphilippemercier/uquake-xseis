#pragma once
// #ifndef CORE_H
// #define CORE_H 

#include <iomanip>
#include <iostream>
#include <vector>
#include <complex>
#include <assert.h>
#include <stdint.h>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>

#define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
#include "gsl/span"


namespace xs {

const uint32_t CACHE_LINE = 64;
// const uint32_t MEM_ALIGNMENT = CACHE_LINE;
const uint32_t MIN_ALIGN = 16;
using Complex32 = std::complex<float>;

template <typename ValueType>
using VecOfSpans = std::vector<gsl::span<ValueType>>; 

using KeyGroups = std::vector<std::vector<uint16_t>>; 

template<typename T>
size_t PadToBytes(const size_t size, const uint32_t nbytes)
{    
	const uint32_t paddingElements = nbytes / sizeof(T);    
	const uint32_t mod = size % paddingElements;
	uint32_t ipad;	
	mod == 0 ? ipad = 0 : ipad = paddingElements - mod;
	return size + ipad;
}

template <typename T>
T* MallocAligned(const size_t N, const uint32_t alignment)
{
	return (T*) aligned_alloc(alignment, N * sizeof(T));
}


template <typename ValueType, typename IndexType=size_t>
class Vector {
public:

	const uint32_t alignment_ = CACHE_LINE;
	using pointer = ValueType*;
	using reference = ValueType&;
	IndexType size_;
	pointer data_;
	bool owns_; // data ownership

	Vector() :data_(nullptr), size_(0), owns_(true) {}

	// init from existing c-array
	Vector(ValueType *data, IndexType size) : data_(data), size_(size), owns_(false) {}

	// init and allocate dynamic memory
	Vector(IndexType size): size_(size), owns_(true)
	{		
		data_ = MallocAligned<ValueType>(size_, alignment_);
	}

	~Vector() { if (owns_==true) free(data_); }

	constexpr gsl::span<ValueType> const span()
	{
		return gsl::make_span(data_, size_);
	}

	constexpr IndexType size() const noexcept { return size_; }
	constexpr pointer data() const noexcept { return data_; }	
	constexpr reference operator[] (IndexType ix) const { return data_[ix]; }
	constexpr pointer begin() const noexcept { return data_;}
	constexpr pointer end() const noexcept {return data_ + size_;}

}; // end Vector


template <typename ValueType, typename IndexType=size_t>
class Array2D {
public:

	const uint32_t alignment_ = CACHE_LINE;
	using pointer = ValueType*;
	using reference = ValueType&;
	IndexType nrow_;
	IndexType ncol_;
	IndexType ncol_pad_;
	// IndexType size_;
	pointer data_;
	bool owns_; // data ownership

	Array2D() :data_(nullptr), nrow_(0), ncol_(0), ncol_pad_(0), owns_(true) {}

	// init from existing c-array, copies data only if padding enabled
	Array2D(ValueType *data, IndexType nrow, IndexType ncol, bool pad)
	: nrow_(nrow), ncol_(ncol)
	{
		if (pad == true)
		{
			ncol_pad_ = PadToBytes<ValueType>(ncol, alignment_);
			data_ = MallocAligned<ValueType>(nrow_ * ncol_pad_, alignment_);
			owns_ = true;
			for(size_t i = 0; i < nrow_; ++i) {
				pointer in = data + i * ncol_;
				pointer out = data_ + i * ncol_pad_;
				std::copy(in, in + ncol_, out);
			}
		}
		else
		{
			data_ = data;
			ncol_pad_ = ncol;
			owns_ = false;
		}
	}
	
	// init and allocate dynamic memory, padded to align each row ptr
	Array2D(IndexType nrow, IndexType ncol): nrow_(nrow), ncol_(ncol), owns_(true)
	{
		ncol_pad_ = PadToBytes<ValueType>(ncol, alignment_);
		data_ = MallocAligned<ValueType>(nrow_ * ncol_pad_, alignment_);
		assert(ncol_pad_ >= ncol);		
	}

	~Array2D() { if (owns_==true) free(data_); }

	VecOfSpans<ValueType> rows()
	{
		VecOfSpans<ValueType> rows;
		rows.reserve(nrow_);

		for(size_t i = 0; i < nrow_; ++i) 
		{
			rows.emplace_back(this->span(i));
			// rows.emplace_back(this->operator()(i));
		}
		return rows;
	}

	constexpr IndexType size_actual() const noexcept { return nrow_ * ncol_pad_; }
	constexpr IndexType size() const noexcept { return nrow_ * ncol_; }
	constexpr IndexType nrow() const noexcept { return nrow_; }
	constexpr IndexType ncol() const noexcept { return ncol_; }
	constexpr pointer data() const noexcept { return data_; }
	constexpr pointer row(IndexType ix_row) const noexcept 
	{
		// assert(ix_row < nrow_);
		return data_ + (ix_row * ncol_pad_);
	}

	// Get value at flattened index ix
	// constexpr reference operator[] (IndexType ix) const { return data_[ix]; }
	// value at ix, iy
	constexpr reference operator() (IndexType ix_row, IndexType ix_col) const { return (row(ix_row) + ix_col)[0]; }
	// constexpr pointer operator() (IndexType ix_row, IndexType ix_col) const { return row(ix_row) + ix_col; }

	// row as view
	// constexpr gsl::span<ValueType> operator() (IndexType ix_row) const
	constexpr gsl::span<ValueType> span(IndexType ix_row) const
	{
		return gsl::make_span(row(ix_row), ncol_);
	}
	// constexpr gsl::span<ValueType> operator[] (IndexType ix_row) const {return gsl::make_span(row(ix_row), ncol_);}


	Vector<ValueType> copy_col(size_t icol) 
	{
		auto vec = Vector<ValueType>(nrow());
		
		for (size_t i = 0; i < nrow(); ++i) {
			vec[i] = this->row(i)[icol];
		}
		
		return vec;	
	}

}; // end Array2D



class Grid {
public:
	// lims_ = {xmin, xmax, ymin, ymax, zmin, zmax, spacing}
	std::vector<float> lims_;
	float spacing;
	float xmin, ymin, zmin;	
	size_t nx, ny, nz;
	float dx, dy, dz;
	size_t npts;


	Grid() {}
	Grid(std::vector<float> lims):
	lims_(lims), spacing(lims_[6]), xmin(lims_[0]), ymin(lims_[2]), zmin(lims_[4]){

		assert(lims.size() == 7);
		float xrange = lims_[1] - lims_[0];
		float yrange = lims_[3] - lims_[2];
		float zrange = lims_[5] - lims_[4];

		nx = std::abs(xrange) / spacing;
		ny = std::abs(yrange) / spacing;
		nz = std::abs(zrange) / spacing;

		dx = ((xrange > 0) - (xrange < 0)) * spacing;
		dy = ((yrange > 0) - (yrange < 0)) * spacing;
		dz = ((zrange > 0) - (zrange < 0)) * spacing;

		npts = static_cast<size_t>(nx * ny * nz);
		printf("Grid (%lu x %lu x %lu) = %lu\n", nx, ny, nz, npts);
	}
	~Grid(){}

	gsl::span<float> lims() { return gsl::make_span(lims_); }


	Array2D<float> points(){

		auto points = Array2D<float>(npts, 3);
		size_t row_ix = 0;
		
		for (size_t i = 0; i < nx; ++i) {
			for (size_t j = 0; j < ny; ++j) {
				for (size_t k = 0; k < nz; ++k) {
					points(row_ix, 0) = xmin + i * dx;
					points(row_ix, 1) = ymin + j * dy;
					points(row_ix, 2) = zmin + k * dz;
					row_ix += 1;
				}			
			}
		}
		return points;
	}	
};



template<typename T>
std::vector<T> Linspace(T start, T stop, size_t size){

		std::vector<T> vec;
		vec.reserve(size);
		float step = (stop - start) / static_cast<float>(size);

		for (size_t i = 0; i < size; ++i) {
			vec.push_back(start + step * static_cast<float>(i));
		}
		return vec;
	}

	
template<typename T>
std::vector<T> Arange(T start, T stop, T step = 1) {
	std::vector<T> values;
	values.reserve((stop - start) / step);
	
	for (T value = start; value < stop; value += step){
			values.push_back(value);
	}

	return values;
}



} // end namespace xs


template<typename T>
std::ostream &operator <<(std::ostream &os, const gsl::span<T> v) {
	os << "[ " ;
	for (auto&& x : v) os << std::left << std::setw(10) << x;
	return os << " ]";
}

template<typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<T> &v) {
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
	return os;
}

template<typename T>
std::ostream &operator <<(std::ostream &os, xs::VecOfSpans<T> vspan) {

	if (vspan.size() < 100000)
	{
		for(auto&& v : vspan) {os << v << "\n";}
	}
	os << "std::vector of gsl::spans (" << vspan.size() << "x" << vspan[0].size() <<")" << "\n";
	// std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
	return os;
}

// #endif

