
template <class Array2D, bool IsConst>
class IteratorArray2D
{
public:	
	using iterator_category = std::random_access_iterator_tag;

	using element_type_ = typename Array2D::value_t;
	// using value_type = std::remove_cv_t<element_type_>;
	using difference_type = typename Array2D::index_t;

	// using reference = std::conditional_t<IsConst, const element_type_, element_type_>&;
	// using pointer = std::add_pointer_t<reference>;

	using value = gsl::span<element_type_>;
	using reference = gsl::span<element_type_>&;
	using pointer = gsl::span<element_type_>*;
	// using reference = std::conditional_t<IsConst, const element_type_, element_type_>&;
	// using pointer = std::add_pointer_t<reference>;

	IteratorArray2D() = default;

	constexpr IteratorArray2D(const Array2D* arr2d, typename Array2D::index_t idx) noexcept
		: arr2d_(arr2d), index_(idx), size_(arr2d->nrow_)
	{}

	friend IteratorArray2D<Array2D, true>;
	template <bool B, std::enable_if_t<!B && IsConst>* = nullptr>
	constexpr IteratorArray2D(const IteratorArray2D<Array2D, B>& other) noexcept
		: IteratorArray2D(other.arr2d_, other.index_)
	{}

	GSL_SUPPRESS(bounds.1) // NO-FORMAT: attribute
	// constexpr reference operator*() const


	reference operator*() 
	{
		Expects(index_ != size_);
		// return *(arr2d_->operator()(index_));
		// return *(arr2d_->view(index_));
		// return *((*arr2d_)(index_));
		return (*arr2d_)(index_);
		// Expects(index_ != size_);
		// return *(arr2d_->data() + index_);
	}

	// pointer operator->() 
	// {
	// 	Expects(index_ != size_);
	// 	return (*arr2d_)(index_);
	// 	// return *(arr2d_->view(index_));
	// 	// Expects(index_ != size_);
	// 	// return arr2d_->data() + index_;

	// }

	constexpr IteratorArray2D& operator++()
	{
		Expects(0 <= index_ && index_ != size_);
		++index_;
		return *this;
	}

	constexpr IteratorArray2D operator++(int)
	{
		auto ret = *this;
		++(*this);
		return ret;
	}

	constexpr IteratorArray2D& operator--()
	{
		Expects(index_ != 0 && index_ <= size_);
		--index_;
		return *this;
	}

	constexpr IteratorArray2D operator--(int)
	{
		auto ret = *this;
		--(*this);
		return ret;
	}

	constexpr IteratorArray2D operator+(difference_type n) const
	{
		auto ret = *this;
		return ret += n;
	}

	friend constexpr IteratorArray2D operator+(difference_type n, IteratorArray2D const& rhs)
	{
		return rhs + n;
	}

	constexpr IteratorArray2D& operator+=(difference_type n)
	{
		Expects((index_ + n) >= 0 && (index_ + n) <= size_);
		index_ += n;
		return *this;
	}

	constexpr IteratorArray2D operator-(difference_type n) const
	{
		auto ret = *this;
		return ret -= n;
	}

	constexpr IteratorArray2D& operator-=(difference_type n) { return *this += -n; }

	constexpr difference_type operator-(IteratorArray2D rhs) const
	{
		Expects(arr2d_ == rhs.arr2d_);
		return index_ - rhs.index_;
	}

	constexpr reference operator[](difference_type n) const { return *(*this + n); }

	constexpr friend bool operator==(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return lhs.arr2d_ == rhs.arr2d_ && lhs.index_ == rhs.index_;
	}

	constexpr friend bool operator!=(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return !(lhs == rhs);
	}

	constexpr friend bool operator<(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return lhs.index_ < rhs.index_;
	}

	constexpr friend bool operator<=(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return !(rhs < lhs);
	}

	constexpr friend bool operator>(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return rhs < lhs;
	}

	constexpr friend bool operator>=(IteratorArray2D lhs, IteratorArray2D rhs) noexcept
	{
		return !(rhs > lhs);
	}

protected:
		const Array2D* arr2d_ = nullptr;
		std::ptrdiff_t index_ = 0;
		size_t size_ = 0;
}; // end a2d iterator