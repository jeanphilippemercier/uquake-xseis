#ifndef ARRAY_H
#define ARRAY_H
#include <stdint.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <assert.h>
// #include <stdlib.h>
// #include <malloc.h>
#include "xseis2/globals.h"

// #define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
#include "gsl/span"


// typedef std::vector<std::vector<IndexType>> VVui64; 
// typedef std::vector<std::vector<uint16_t>> VVui16; 	


namespace xseis {

template <typename ValueType>
using VecOfSpans = std::vector<gsl::span<ValueType>>; 

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
class Array2D {
public:

	const uint32_t alignment_ = MEM_ALIGNMENT;
	using pointer = ValueType*;
	using reference = ValueType&;
	IndexType nrow_;
	IndexType ncol_;
	IndexType ncol_pad_;
	// IndexType size_;
	pointer data_;
	bool owns_; // data ownership

	Array2D() :data_(nullptr), nrow_(0), ncol_(0), ncol_pad_(0), owns_(false) {}

	// // init from existing c-array
	// Array2D(ValueType *data, IndexType nrow, IndexType ncol, IndexType npad=0)
	// : data_(data), nrow_(nrow), ncol_(ncol), ncol_pad_(ncol - npad), owns_(false)
	// {
	// 	assert(ncol > npad);
	// }

	// init and allocate dynamic memory
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
		assert(ix_row < nrow_);
		return data_ + (ix_row * ncol_pad_);
	}

	// Get value at flattened index ix
	// constexpr reference operator[] (IndexType ix) const { return data_[ix]; }
	// value at ix, iy
	constexpr reference operator() (IndexType ix_row, IndexType ix_col) const { return (row(ix_row) + ix_col)[0]; }
	// row as view
	// constexpr gsl::span<ValueType> operator() (IndexType ix_row) const
	constexpr gsl::span<ValueType> span(IndexType ix_row) const
	{
		return gsl::make_span(row(ix_row), ncol_);
	}

}; // end Array2D


template <typename ValueType, typename IndexType=size_t>
class Vector {
public:

	const uint32_t alignment_ = MEM_ALIGNMENT;
	using pointer = ValueType*;
	using reference = ValueType&;
	IndexType size_;
	pointer data_;
	bool owns_; // data ownership

	Vector() :data_(nullptr), size_(0), owns_(false) {}

	// init from existing c-array
	Vector(ValueType *data, IndexType size) : data_(data), size_(size), owns_(false) {}

	// init and allocate dynamic memory
	Vector(IndexType size): size_(size)
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



} // end namespace xseis


template<typename T>
std::ostream &operator <<(std::ostream &os, const gsl::span<T> &v) {
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
std::ostream &operator <<(std::ostream &os, xseis::Array2D<T> &arr) {

	os << "Array2D (" << arr.nrow() << "x" << arr.ncol() <<")" << "\n";

	if (arr.size() < 1000000)
	{
		for(auto&& v : arr.rows()) {os << v << "\n";}
	}
	// std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
	return os;
}


#endif
