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
    
    template <typename T, typename Compare>
    constexpr const T& clamp(T const& value, T const& lo, T const& hi, Compare comp) {
        return comp(value, hi) ? std::max(value, lo, comp) :
                                 std::min(value, hi, comp);
    }
    
    template <typename T>
    constexpr const T& clamp(T const& value, T const& lo, T const& hi) {
        return detail::clamp(value, lo, hi, std::less<>());
    }
    
}

namespace valarray {
    
    /// auto bytes = valarray::cast<byte>(floatva);
    
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
    
    /// auto bytes = valarray::cast<byte>(floatva, 0x00);
    
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
    
    /// auto bytes = valarray::cast<byte>(floatva, [](float const& v) -> byte {
    ///     return static_cast<byte>(
    ///         detail::clamp(v * 255.0, 0.00, 255.00));
    /// });
    
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
    
    /// auto bytes = valarray::cast<byte>(floatva, [](float v) -> byte {
    ///     return static_cast<byte>(
    ///         detail::clamp(v * 255.0, 0.00, 255.00));
    /// });
    
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
