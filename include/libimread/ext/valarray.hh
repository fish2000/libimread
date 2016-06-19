/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_VALARRAY_HH_
#define LIBIMREAD_EXT_VALARRAY_HH_

#include <valarray>
#include <iterator>
#include <algorithm>
#include <utility>

namespace detail {

    /// pollyfills for C++17 std::clamp()
    /// q.v. http://ideone.com/IpcDt9, http://en.cppreference.com/w/cpp/algorithm/clamp
    template <class T, class Compare>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp) {
        return comp(v, hi) ? std::max(v, lo, comp) : std::min(v, hi, comp);
    }
    template <class T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
        return clamp(v, lo, hi, std::less<>());
    }

}

namespace valarray {
    
    template <typename T, typename U,
              template <typename, typename...>
              class container = std::valarray>
    std::valarray<T> cast(container<U> const& orig) {
        std::valarray<T> out(orig.size());
        std::valarray<U> in(&orig[0], orig.size());
        std::transform(std::begin(in), std::end(in),
                       std::begin(out),
                    [](U const& u) -> T { return T(u); });
        return out;
    }
    
    template <typename T, typename U,
              template <typename, typename...>
              class container = std::valarray>
    std::valarray<T> cast(container<U> const& orig, T const& initial) {
        std::valarray<T> out(initial, orig.size());
        std::valarray<U> in(&orig[0], orig.size());
        std::transform(std::begin(in), std::end(in),
                       std::begin(out),
                    [](U const& u) -> T { return T(u); });
        return out;
    }
    
    template <typename T, typename U>
    using castref_t = std::add_pointer_t<T(U const&)>;
    
    template <typename T, typename U>
    using castval_t = std::add_pointer_t<T(U)>;
    
    template <typename T, typename U,
              template <typename, typename...>
              class container = std::valarray>
    std::valarray<T> cast(container<U> const& orig, castref_t<T, U> caster) {
        std::valarray<T> out(orig.size());
        std::valarray<U> in(&orig[0], orig.size());
        std::transform(std::begin(in), std::end(in),
                       std::begin(out),
                       caster);
        return out;
    }
    
    template <typename T, typename U,
              template <typename, typename...>
              class container = std::valarray>
    std::valarray<T> cast(container<U> const& orig, castval_t<T, U> caster) {
        std::valarray<T> out(orig.size());
        std::valarray<U> in(&orig[0], orig.size());
        std::transform(std::begin(in), std::end(in),
                       std::begin(out),
                       caster);
        return out;
    }
    
}

#endif /// LIBIMREAD_EXT_VALARRAY_HH_
