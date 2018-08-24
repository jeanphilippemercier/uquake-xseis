#ifndef ARRAY2D_H
#define ARRAY2D_H
#include <stdint.h>
#include <iomanip>
#include <iostream>
#include <vector>
// #include <stdlib.h>
// #include <malloc.h>

// #define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
#include "gsl/span"


// typedef std::vector<std::vector<IndexType>> VVui64; 
// typedef std::vector<std::vector<uint16_t>> VVui16; 	

namespace xseis {


// const uint32_t CACHE_LINE = 64;
// using Pad64 = const uint32_t 64;

const uint32_t CACHE_LINE = 64;
// const uint32_t MEM_ALIGNMENT = 16;
const uint32_t MEM_ALIGNMENT = CACHE_LINE;


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


template <typename ScalarType, typename IndexType=size_t>
class Array2D {
public:

	const uint32_t alignment_ = MEM_ALIGNMENT;
	using pointer = ScalarType*;
	using reference = ScalarType&;
	IndexType nrow_;
	IndexType ncol_;
	IndexType ncol_pad_;
	// IndexType size_;
	pointer data_;
	bool owns_; // data ownership

	Array2D() :data_(nullptr), nrow_(0), ncol_(0), ncol_pad_(0), owns_(false) {}

	// init from existing c-array
	Array2D(ScalarType *data, IndexType nrow, IndexType ncol, IndexType npad=0)
	: data_(data), nrow_(nrow), ncol_(ncol), ncol_pad_(ncol - npad), owns_(false)
	{
		assert(ncol > npad);
	}

	// init and allocate dynamic memory
	Array2D(IndexType nrow, IndexType ncol): nrow_(nrow), ncol_(ncol), owns_(true)
	{
		ncol_pad_ = PadToBytes<ScalarType>(ncol, alignment_);
		data_ = MallocAligned<ScalarType>(nrow_ * ncol_pad_, alignment_);
		assert(ncol_pad_ >= ncol);		
	}

	~Array2D() { if (owns_==true) free(data_); }

	std::vector<gsl::span<ScalarType>> rows()
	{
		std::vector<gsl::span<ScalarType>> rows;
		rows.reserve(nrow_);

		for(size_t i = 0; i < nrow_; ++i) 
		{
			rows.emplace_back(this->operator()(i));
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
	constexpr gsl::span<ScalarType> operator() (IndexType ix_row) const
	{
		return gsl::make_span(row(ix_row), ncol_);
	}

}; // end Array2D


} // end namespace xseis


template<typename T>
std::ostream &operator <<(std::ostream &os, const gsl::span<T> &v) {
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
	return os;
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


// template <typename T, std::size_t N = 16>
// class AlignmentAllocator {
// public:
//   typedef T value_type;
//   typedef std::size_t size_type;
//   typedef std::ptrdiff_t difference_type;

//   typedef T * pointer;
//   typedef const T * const_pointer;

//   typedef T & reference;
//   typedef const T & const_reference;

//   public:
//   inline AlignmentAllocator () throw () { }

//   template <typename T2>
//   inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }

//   inline ~AlignmentAllocator () throw () { }

//   inline pointer adress (reference r) {
//     return &r;
//   }

//   inline const_pointer adress (const_reference r) const {
//     return &r;
//   }

//   inline pointer allocate (size_type n) {
//      return (pointer) aligned_alloc(N, n * sizeof(value_type));
//   }

//   inline void deallocate (pointer p, size_type) {
//     _aligned_free (p);
//   }

//   inline void construct (pointer p, const value_type & wert) {
//      new (p) value_type (wert);
//   }

//   inline void destroy (pointer p) {
//     p->~value_type ();
//   }

//   inline size_type max_size () const throw () {
//     return size_type (-1) / sizeof (value_type);
//   }

//   template <typename T2>
//   struct rebind {
//     typedef AlignmentAllocator<T2, N> other;
//   };

//   bool operator!=(const AlignmentAllocator<T,N>& other) const  {
//     return !(*this == other);
//   }

//   // Returns true if and only if storage allocated from *this
//   // can be deallocated from other, and vice versa.
//   // Always returns true for stateless allocators.
//   bool operator==(const AlignmentAllocator<T,N>& other) const {
//     return true;
//   }
// };




#endif
