/*
 * array_view -- https://github.com/wardw/array_view -- by Tom Ward (wardw)
 * 
 * An implementation of the ISO C++ proposal [_Multidimensional bounds, offset
 * and array_view_][1] by Mendakiewicz & Sutter.  This implements the original
 * proposal [N3851][1] up to the last [revision 7][3].  As I understand it,
 * the proposal was with an aim for inclusion in C++17 although is no longer
 * under consideration in its current form.  For later contributions, see
 * the successive proposal [p0122r0][4] (and related [GSL][5] implementation),
 * and alternative proposals such as [P0546][6].
 * 
 * This implementation follows the original proposal by Mendakiewicz & Sutter
 * and subsequent revisions ([latest revision 7][7]). As noted in the proposal,
 * the proposal itself builds on previous work by Callahan, Levanoni and Sutter
 * for their work designing the original interfaces for C++ AMP. Minor post-hoc
 * [additions][8] by Alexander Böhn expand on, and slightly depart from, these
 * design documents.
 * 
 * [1]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3851.pdf
 * [2]: https://msdn.microsoft.com/en-us/library/hh265137.aspx
 * [3]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4512.html
 * [4]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0122r0.pdf
 * [5]: https://github.com/Microsoft/GSL
 * [6]: https://github.com/kokkos/array_ref/blob/master/proposals/P0546.rst
 * [7]: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4512.html
 * [8]: https://git.io/vFNCY
 *
 * BSD 2-clause “Simplified” License
 * 
 * Copyright (c) 2015, Tom Ward
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * + Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * + Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#pragma once

#include <initializer_list>
#include <iostream>
#include <iterator>
#include <array>
#include <cassert>
#include <cstdint>

namespace av {
    
    template <std::size_t Rank>
    class offset;
    
    template <std::size_t Rank>
    class bounds;
    
    template <std::size_t Rank>
    class bounds_iterator;
    
    template <typename T, std::size_t Rank>
    class array_view;
    
    template <typename T, std::size_t Rank>
    class strided_array_view;
    
    template <std::size_t Rank>
    class offset {
        public:
            // constants and types
            static constexpr std::size_t rank = Rank;
            using reference              = std::ptrdiff_t&;
            using const_reference        = std::ptrdiff_t const&;
            using size_type              = std::size_t;
            using value_type             = std::ptrdiff_t;
            
            static_assert(Rank > 0, "Size of Rank must be greater than 0");
            
            // construction
            constexpr offset() noexcept {}
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr offset(value_type v) noexcept {
                (*this)[0] = v;
            }
            constexpr offset(std::initializer_list<value_type> il);
            
            // element access
            constexpr reference       operator[](size_type n) { return offset_[n]; }
            constexpr const_reference operator[](size_type n) const { return offset_[n]; }
            
            // arithmetic
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr offset& operator++() {
                return ++(*this)[0];
            }
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr offset operator++(int) {
                return offset<Rank>{ (*this)[0]++ };
            }
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr offset& operator--() {
                return --(*this)[0];
            }
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr offset operator--(int) {
                return offset<Rank>{ (*this)[0]-- };
            }
            
            constexpr offset& operator+=(offset const& rhs);
            constexpr offset& operator-=(offset const& rhs);
            
            constexpr offset operator+() const noexcept { return *this; }
            constexpr offset operator-() const {
                offset<Rank> copy{ *this };
                for (value_type& elem : copy.offset_) {
                    elem *= -1;
                }
                return copy;
            }
            
            constexpr offset& operator*=(value_type v);
            constexpr offset& operator/=(value_type v);
            
            constexpr offset transpose() const noexcept;
            
        private:
            std::array<value_type, rank> offset_ = {};
    };
    
    template <std::size_t Rank>
    constexpr offset<Rank>::offset(std::initializer_list<value_type> il) {
        // Note `il` is not a constant expression, hence the runtime assert for now
        assert(il.size() == Rank);
        std::copy(il.begin(), il.end(), offset_.data());
    }
    
    // arithmetic
    template <std::size_t Rank>
    constexpr offset<Rank>& offset<Rank>::operator+=(offset const& rhs) {
        for (size_type i = 0; i < Rank; ++i) {
            (*this)[i] += rhs[i];
        }
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank>& offset<Rank>::operator-=(offset const& rhs) {
        for (size_type i = 0; i < Rank; ++i) {
            (*this)[i] -= rhs[i];
        }
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank>& offset<Rank>::operator*=(value_type v) {
        for (value_type& elem : offset_) {
            elem *= v;
        }
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank>& offset<Rank>::operator/=(value_type v) {
        for (value_type& elem : offset_) {
            elem /= v;
        }
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank> offset<Rank>::transpose() const noexcept {
        offset<Rank> transposed{ *this };
        std::reverse_copy(std::begin(offset_),
                          std::end(offset_),
                          std::begin(transposed.offset_));
        return transposed;
    }
    
    // Free functions
    
    // offset equality
    template <std::size_t Rank>
    constexpr bool operator==(offset<Rank> const& lhs, offset<Rank> const& rhs) noexcept {
        for (std::size_t i = 0; i < Rank; ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
    
    template <std::size_t Rank>
    constexpr bool operator!=(offset<Rank> const& lhs, offset<Rank> const& rhs) noexcept {
        return !(lhs == rhs);
    }
    
    // offset arithmetic
    template <std::size_t Rank>
    constexpr offset<Rank> operator+(offset<Rank> const& lhs, offset<Rank> const& rhs) {
        return offset<Rank>{ lhs } += rhs;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank> operator-(offset<Rank> const& lhs, offset<Rank> const& rhs) {
        return offset<Rank>{ lhs } -= rhs;
    }

    template <std::size_t Rank>
    constexpr offset<Rank> operator*(offset<Rank> const& lhs, std::ptrdiff_t v) {
        return offset<Rank>{ lhs } *= v;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank> operator*(std::ptrdiff_t v, offset<Rank> const& rhs) {
        return offset<Rank>{ rhs } *= v;
    }
    
    template <std::size_t Rank>
    constexpr offset<Rank> operator/(offset<Rank> const& lhs, std::ptrdiff_t v) {
        return offset<Rank>{ lhs } /= v;
    }
    
    template <std::size_t Rank>
    class bounds {
        public:
            // constants and types
            static constexpr std::size_t rank = Rank;
            using reference              = std::ptrdiff_t&;
            using const_reference        = std::ptrdiff_t const&;
            using iterator               = bounds_iterator<Rank>;
            using const_iterator         = bounds_iterator<Rank>;
            using size_type              = std::size_t;
            using value_type             = std::ptrdiff_t;
        
            static_assert(Rank > 0, "Size of Rank must be greater than 0");
        
            // construction
            constexpr bounds() noexcept {};
            
            // Question: is there a reason this constructor is not `noexcept` ?
            template <std::size_t R = Rank, typename = std::enable_if_t<R == 1>>
            constexpr bounds(value_type v) {
                (*this)[0] = v;
                postcondition();
            }
            constexpr bounds(std::initializer_list<value_type> il);
            
            // observers
            constexpr size_type size() const noexcept;
            constexpr bool      contains(offset<Rank> const& idx) const noexcept;
            
            // iterators
            const_iterator begin() const noexcept { return const_iterator{ *this }; };
            const_iterator end() const noexcept {
                iterator iter{ *this };
                return iter._setOffTheEnd();
            }
            
            // element access
            constexpr reference       operator[](size_type n) { return bounds_[n]; }
            constexpr const_reference operator[](size_type n) const { return bounds_[n]; }
            
            // arithmetic
            constexpr bounds& operator+=(offset<Rank> const& rhs);
            constexpr bounds& operator-=(offset<Rank> const& rhs);
            
            constexpr bounds& operator*=(value_type v);
            constexpr bounds& operator/=(value_type v);
            
            constexpr bounds transpose() const noexcept;
            
        private:
            std::array<value_type, rank> bounds_ = {};
            void postcondition() {
                /* todo */
            };
    };
    
    // construction
    template <std::size_t Rank>
    constexpr bounds<Rank>::bounds(const std::initializer_list<value_type> il) {
        assert(il.size() == Rank);
        std::copy(il.begin(), il.end(), bounds_.data());
        postcondition();
    }
    
    // observers
    template <std::size_t Rank>
    constexpr std::size_t bounds<Rank>::size() const noexcept {
        size_type product{ 1 };
        for (const value_type& elem : bounds_) {
            product *= elem;
        }
        return product;
    }
    
    template <std::size_t Rank>
    constexpr bool bounds<Rank>::contains(offset<Rank> const& idx) const noexcept {
        for (size_type i = 0; i < Rank; ++i) {
            if (!(0 <= idx[i] && idx[i] < (*this)[i])) {
                return false;
            }
        }
        return true;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank> bounds<Rank>::transpose() const noexcept {
        bounds<Rank> transposed{ *this };
        std::reverse_copy(std::begin(bounds_),
                          std::end(bounds_),
                          std::begin(transposed.bounds_));
        transposed.postcondition();
        return transposed;
    }
    
    
    // iterators
    // todo
    
    // arithmetic
    template <std::size_t Rank>
    constexpr bounds<Rank>& bounds<Rank>::operator+=(offset<Rank> const& rhs) {
        for (size_type i = 0; i < Rank; ++i) {
            bounds_[i] += rhs[i];
        }
        postcondition();
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank>& bounds<Rank>::operator-=(offset<Rank> const& rhs) {
        for (size_type i = 0; i < Rank; ++i) {
            bounds_[i] -= rhs[i];
        }
        postcondition();
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank>& bounds<Rank>::operator*=(value_type v) {
        for (value_type& elem : bounds_) {
            elem *= v;
        }
        postcondition();
        return *this;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank>& bounds<Rank>::operator/=(value_type v) {
        for (value_type& elem : bounds_) {
            elem /= v;
        }
        postcondition();
        return *this;
    }
    
    // Free functions
    
    // bounds equality
    template <std::size_t Rank>
    constexpr bool operator==(bounds<Rank> const& lhs, bounds<Rank> const& rhs) noexcept {
        for (std::size_t i = 0; i < Rank; ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
    
    template <std::size_t Rank>
    constexpr bool operator!=(bounds<Rank> const& lhs, bounds<Rank> const& rhs) noexcept {
        return !(lhs == rhs);
    }
    
    // bounds arithmetic
    template <std::size_t Rank>
    constexpr bounds<Rank> operator+(bounds<Rank> const& lhs, offset<Rank> const& rhs) {
        return bounds<Rank>{ lhs } += rhs;
    }

    template <std::size_t Rank>
    constexpr bounds<Rank> operator+(offset<Rank> const& lhs, bounds<Rank> const& rhs) {
        return bounds<Rank>{ rhs } += lhs;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank> operator-(bounds<Rank> const& lhs, offset<Rank> const& rhs) {
        return bounds<Rank>{ lhs } -= rhs;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank> operator*(bounds<Rank> const& lhs, std::ptrdiff_t v) {
        return bounds<Rank>{ lhs } *= v;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank> operator*(std::ptrdiff_t v, bounds<Rank> const& rhs) {
        return bounds<Rank>{ rhs } *= v;
    }
    
    template <std::size_t Rank>
    constexpr bounds<Rank> operator/(bounds<Rank> const& lhs, std::ptrdiff_t v) {
        return bounds<Rank>{ lhs } /= v;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> begin(bounds<Rank> const& b) noexcept {
        return b.begin();
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> end(bounds<Rank> const& b) noexcept {
        return b.end();
    }
    
    template <std::size_t Rank>
    class bounds_iterator {
        public:
            using iterator_category = std::random_access_iterator_tag; // unspecified but satisfactory
            using value_type        = offset<Rank>;
            using difference_type   = std::ptrdiff_t;
            using pointer           = offset<Rank>*; // unspecified but satisfactory (?)
            using reference         = const offset<Rank>;
            
            static_assert(Rank > 0, "Size of Rank must be greater than 0");
            
            bounds_iterator(const bounds<Rank> bounds, offset<Rank> off = offset<Rank>()) noexcept
                : bounds_(bounds)
                , offset_(off) {}
            
            bool operator==(bounds_iterator const& rhs) const {
                // Requires *this and rhs are iterators over the same bounds object.
                return offset_ == rhs.offset_;
            }
            
            bounds_iterator& operator++();
            bounds_iterator  operator++(int);
            bounds_iterator& operator--();
            bounds_iterator  operator--(int);
            
            bounds_iterator  operator+(difference_type n) const;
            bounds_iterator& operator+=(difference_type n);
            bounds_iterator  operator-(difference_type n) const;
            bounds_iterator& operator-=(difference_type n);
            
            difference_type operator-(bounds_iterator const& rhs) const;
            
            // Note this iterator is not a true random access iterator, nor meets N4512
            // + operator* returns a value type (and not a reference)
            // + operator-> returns a pointer to the current value type, which breaks N4512 as this
            //   must be considered invalidated after any subsequent operation on this iterator
            reference operator*()  const { return offset_; }
            pointer   operator->() const { return &offset_; }
            
            reference operator[](difference_type n) const {
                bounds_iterator<Rank> iter(*this);
                return (iter += n).offset_;
            }
            
            bounds_iterator& _setOffTheEnd();
            
            bounds_iterator transpose() const noexcept;
            
        private:
            bounds<Rank> bounds_;
            offset<Rank> offset_;
    };
    
    template <std::size_t Rank>
    bounds_iterator<Rank> bounds_iterator<Rank>::operator++(int) {
        bounds_iterator tmp(*this);
        ++(*this);
        return tmp;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank>& bounds_iterator<Rank>::operator++() {
        // watchit: dim must be signed in order to fail the condition dim>=0
        for (int dim = (Rank - 1); dim >= 0; --dim) {
            if (++offset_[dim] < bounds_[dim]) {
                return (*this);
            } else {
                offset_[dim] = 0;
            }
        }
        // off-the-end value
        _setOffTheEnd();
        return *this;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank>& bounds_iterator<Rank>::operator--() {
        // watchit: dim must be signed in order to fail the condition dim>=0
        for (int dim = (Rank - 1); dim >= 0; --dim) {
            if (--offset_[dim] >= 0) {
                return (*this);
            } else {
                offset_[dim] = bounds_[dim] - 1;
            }
        }
        // before-the-start value
        for (int dim = 0; dim < Rank - 1; ++dim) {
            offset_[dim] = 0;
        }
        offset_[Rank - 1] = -1;
        return *this;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> bounds_iterator<Rank>::operator--(int) {
        bounds_iterator tmp(*this);
        --(*this);
        return tmp;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank>& bounds_iterator<Rank>::_setOffTheEnd() {
        for (std::size_t dim = 0; dim < Rank - 1; ++dim) {
            offset_[dim] = bounds_[dim] - 1;
        }
        offset_[Rank - 1] = bounds_[Rank - 1];
        return *this;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> bounds_iterator<Rank>::transpose() const noexcept {
        return bounds_iterator(bounds_.transpose(),
                               offset_.transpose());
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank>& bounds_iterator<Rank>::operator+=(difference_type n) {
        for (int dim = (Rank - 1); dim >= 0; --dim) {
            difference_type remainder = (n + offset_[dim]) % bounds_[dim];
            n                         = (n + offset_[dim]) / bounds_[dim];
            offset_[dim]              = remainder;
        }
        assert(n == 0); // no overflow
        return *this;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> bounds_iterator<Rank>::operator+(difference_type n) const {
        bounds_iterator<Rank> iter(*this);
        return iter += n;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank>& bounds_iterator<Rank>::operator-=(difference_type n) {
        // take (diminished) radix compliment
        auto diminishedRadixComplement = [&]() {
            for (int dim = (Rank - 1); dim >= 0; --dim) {
                offset_[dim] = bounds_[dim] - offset_[dim];
            }
        };
        diminishedRadixComplement();
        *this += n;
        diminishedRadixComplement();
        return *this;
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> bounds_iterator<Rank>::operator-(difference_type n) const {
        bounds_iterator<Rank> iter(*this);
        return iter -= n;
    }
    
    // Free functions
    
    template <std::size_t Rank>
    bool operator==(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return lhs.operator==(rhs);
    }
    
    template <std::size_t Rank>
    bool operator!=(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return !lhs.operator==(rhs);
    }
    
    template <std::size_t Rank>
    bool operator<(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return rhs - lhs > 0;
    }
    
    template <std::size_t Rank>
    bool operator<=(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return !(lhs > rhs);
    }
    
    template <std::size_t Rank>
    bool operator>(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return rhs < lhs;
    }
    
    template <std::size_t Rank>
    bool operator>=(bounds_iterator<Rank> const& lhs, bounds_iterator<Rank> const& rhs) {
        return !(lhs < rhs);
    }
    
    template <std::size_t Rank>
    bounds_iterator<Rank> operator+(typename bounds_iterator<Rank>::difference_type n,
                                    bounds_iterator<Rank> const&                    rhs);
    
    namespace {
        
        template <typename Viewable, typename U, typename View = std::remove_reference_t<Viewable>>
        using is_viewable_on_u = std::integral_constant<bool,
            std::is_convertible<typename View::size_type, std::ptrdiff_t>::value &&
                std::is_convertible<typename View::value_type*, std::add_pointer_t<U>>::value &&
                std::is_same<std::remove_cv_t<typename View::value_type>,
                             std::remove_cv_t<U>>::value>;
        
        template <typename T, typename U>
        using is_viewable_value = std::integral_constant<
            bool, std::is_convertible<std::add_pointer_t<T>, std::add_pointer_t<U>>::value &&
                      std::is_same<std::remove_cv_t<T>, std::remove_cv_t<U>>::value>;
        
        template <typename T, std::size_t Rank>
        constexpr T& view_access(T* data, offset<Rank> const& idx, offset<Rank> const& stride) {
            std::ptrdiff_t off{};
            for (std::size_t i = 0; i < Rank; ++i) {
                off += idx[i] * stride[i];
            }
            return data[off];
        }
        
    } // namespace
    
    template <typename T, std::size_t Rank = 1>
    class array_view {
        public:
            static constexpr std::size_t rank = Rank;
            using offset_type            = offset<Rank>;
            using bounds_type            = class bounds<Rank>;
            using size_type              = std::size_t;
            using value_type             = T;
            using pointer                = T*;
            using reference              = T&;
            
            static_assert(Rank > 0, "Size of Rank must be greater than 0");
            
            constexpr array_view() noexcept
                : data_(nullptr) {}
            
            template <typename Viewable, std::size_t R = Rank,
                      typename = std::enable_if_t<
                          R == 1 && is_viewable_on_u<Viewable, value_type>::value
                          // todo: && decay_t<Viewable> is not a specialization of array_view
                          >>
            // todo: assert static_cast<U*>(vw.data()) points to contiguous data of at least vw.size()
            constexpr array_view(Viewable&& vw)
                : data_(vw.data())
                , bounds_(vw.size()) {}
                      
            template <typename U, std::size_t R = Rank,
                      typename = std::enable_if_t<R == 1 && is_viewable_value<U, value_type>::value>>
            constexpr array_view(array_view<U, R> const& rhs) noexcept
                : data_(rhs.data())
                , bounds_(rhs.bounds()) {}
            
            template <std::size_t Extent, typename = std::enable_if_t<Extent == 1>>
            constexpr array_view(value_type (&arr)[Extent]) noexcept
                : data_(arr)
                , bounds_(Extent) {}
            
            template <typename U, typename = std::enable_if_t<is_viewable_value<U, value_type>::value>>
            constexpr array_view(array_view<U, Rank> const& rhs) noexcept
                : data_(rhs.data())
                , bounds_(rhs.bounds()) {}
            
            template <typename Viewable,
                      typename = std::enable_if_t<is_viewable_on_u<Viewable, value_type>::value>>
            constexpr array_view(Viewable&& vw, bounds_type bounds)
                : data_(vw.data())
                , bounds_(bounds) {
                assert(bounds.size() <= vw.size());
            }
            
            constexpr array_view(pointer ptr, bounds_type bounds)
                : data_(ptr)
                , bounds_(bounds) {}
            
            // observers
            constexpr bounds_type bounds() const noexcept { return bounds_; }
            constexpr size_type   size() const noexcept { return bounds().size(); }
            constexpr offset_type stride() const noexcept;
            constexpr pointer     data() const noexcept { return data_; }
            
            constexpr reference operator[](offset_type const& idx) const {
                assert(bounds().contains(idx) == true);
                return view_access(data_, idx, stride());
            }
            
            // slicing and sectioning
            template <std::size_t R = Rank, typename = std::enable_if_t<R >= 2>>
            constexpr array_view<T, Rank - 1> operator[](std::ptrdiff_t slice) const {
                assert(0 <= slice && slice < bounds()[0]);
                av::bounds<Rank - 1> new_bounds{};
                for (std::size_t i = 0; i < rank - 1; ++i) {
                    new_bounds[i] = bounds()[i + 1];
                }
                std::ptrdiff_t off = slice * stride()[0];
                return array_view<T, Rank - 1>(data_ + off, new_bounds);
            }
            
            constexpr strided_array_view<T, Rank> section(offset_type const& origin,
                                                          bounds_type const& section_bounds) const {
                // todo: requirement is for any idx in section_bounds (boundary fail)
                // assert(bounds().contains(origin + section_bounds) == true);
                return strided_array_view<T, Rank>(&(*this)[origin], section_bounds, stride());
            }
            
            constexpr strided_array_view<T, Rank> section(offset_type const& origin) const {
                // todo: requires checking for any idx in bounds() - origin
                // assert(bounds().contains(bounds()) == true);
                return strided_array_view<T, Rank>(&(*this)[origin], bounds() - origin, stride());
            }
            
            constexpr array_view<T, Rank> transpose() const noexcept {
                return array_view<T, Rank>(*this, bounds_.transpose()); 
            }
            
        private:
            pointer     data_;
            bounds_type bounds_;
    };
    
    template <typename T, std::size_t Rank>
    constexpr typename array_view<T, Rank>::offset_type array_view<T, Rank>::stride() const noexcept {
        offset_type stride{};
        stride[rank - 1] = 1;
        for (int dim = static_cast<int>(rank) - 2; dim >= 0; --dim) {
            stride[dim] = stride[dim + 1] * bounds()[dim + 1];
        }
        return stride;
    }
    
    template <class T, std::size_t Rank = 1>
    class strided_array_view {
        public:
            // constants and types
            static constexpr std::size_t rank = Rank;
            using offset_type            = offset<Rank>;
            using bounds_type            = class bounds<Rank>;
            using size_type              = std::size_t;
            using value_type             = T;
            using pointer                = T*;
            using reference              = T&;
            
            // constructors, copy, and assignment
            constexpr strided_array_view() noexcept
                : data_{ nullptr }
                , bounds_{}
                , stride_{} {}
            
            template <typename U, typename = std::enable_if_t<is_viewable_value<U, value_type>::value>>
            constexpr strided_array_view(array_view<U, Rank> const& rhs) noexcept
                : data_{ rhs.data() }
                , bounds_{ rhs.bounds() }
                , stride_{ rhs.stride() } {}
            
            template <typename U, typename = std::enable_if_t<is_viewable_value<U, value_type>::value>>
            constexpr strided_array_view(strided_array_view<U, Rank> const& rhs) noexcept
                : data_{ rhs.data_ }
                , bounds_{ rhs.bounds() }
                , stride_{ rhs.stride() } {}
            
            constexpr strided_array_view(pointer ptr, bounds_type bounds, offset_type stride)
                : data_(ptr)
                , bounds_(bounds)
                , stride_(stride) {
                // todo: assert that sum(idx[i] * stride[i]) fits in std::ptrdiff_t
            }
            
            // observers
            constexpr bounds_type bounds() const noexcept { return bounds_; }
            constexpr size_type   size() const noexcept { return bounds_.size(); }
            constexpr offset_type stride() const noexcept { return stride_; }
            
            // element access
            constexpr reference operator[](offset_type const& idx) const {
                assert(bounds().contains(idx) == true);
                return view_access(data_, idx, stride_);
            }
        
            // slicing and sectioning
            template <std::size_t R = Rank, typename = std::enable_if_t<R >= 2>>
            constexpr strided_array_view<T, Rank - 1> operator[](std::ptrdiff_t slice) const {
                assert(0 <= slice && slice < bounds()[0]);
                av::bounds<Rank - 1> new_bounds{};
                for (std::size_t i = 0; i < rank - 1; ++i) {
                    new_bounds[i] = bounds()[i + 1];
                }
                av::offset<Rank - 1> new_stride{};
                for (std::size_t i = 0; i < rank - 1; ++i) {
                    new_stride[i] = stride()[i + 1];
                }
                std::ptrdiff_t off = slice * stride()[0];
                return strided_array_view<T, Rank - 1>(data_ + off, new_bounds, new_stride);
            }
            
            constexpr strided_array_view<T, Rank> section(offset_type const& origin,
                                                          bounds_type const& section_bounds) const {
                // todo: requirement is for any idx in section_bounds (boundary fail)
                // assert(bounds().contains(origin + section_bounds) == true);
                return strided_array_view<T, Rank>(&(*this)[origin], section_bounds, stride());
            }
            
            constexpr strided_array_view<T, Rank> section(offset_type const& origin) const {
                // todo: requires checking for any idx in bounds() - origin
                // assert(bounds().contains(bounds()) == true);
                return strided_array_view<T, Rank>(&(*this)[origin], bounds() - origin, stride());
            }
            
            constexpr strided_array_view<T, Rank> transpose() const noexcept {
                return strided_array_view<T, Rank>(data_, bounds_.transpose(),
                                                          stride_.transpose()); 
            }
            
        private:
            pointer     data_;
            bounds_type bounds_;
            offset_type stride_;
    };
    
    template <std::size_t Rank>
    std::ostream& operator<<(std::ostream& os, av::offset<Rank> const& off) {
        /// av::offset<…> output stream print helper
        /// reproduced verbatim, more or less, from original testsuite
        os << "(" << off[0];
        for (std::size_t i = 1; i < off.rank; ++i) {
            os << "," << off[i];
        }
        return os << ")";
    }
    
    template <std::size_t Rank>
    std::ostream& operator<<(std::ostream& os, av::bounds<Rank> const& bnd) {
        /// av::bounds<…> output stream print helper
        /// adapted from av::offset<…> stream print helper
        /// from original testsuite
        os << "{" << bnd[0];
        for (std::size_t i = 1; i < bnd.rank; ++i) {
            os << "," << bnd[i];
        }
        return os << "}";
    }
    
} // namespace av
