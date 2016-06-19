/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HISTOGRAM_HH_
#define LIBIMREAD_HISTOGRAM_HH_

#include <iterator>
#include <functional>
#include <type_traits>
#include <algorithm>

#include <libimread/libimread.hpp>

namespace im {
    
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
    
    /// forward-declare im::Image
    class Image;
    
    class Histogram {
        
        using begin_t = decltype(std::begin(std::declval<std::valarray>(UCHAR_MAX)));
        using   end_t = decltype(  std::end(std::declval<std::valarray>(UCHAR_MAX)));
        
        public:
            explicit Histogram(Image*);
            
            std::size_t size() const;
            begin_t begin();
            end_t end();
            float sum() const;
            float min() const;
            float max() const;
            float entropy() const;
            std::valarray<byte> const& sourcedata() const;
            std::valarray<float>& values();
            std::valarray<float> const& values() const;
        
        protected:
            float flinitial = 0.00000000;
            float entropy = 0.0;
            bool entropy_calculated = false;
            std::valarray<byte> data;
            std::valarray<float> histogram(std::cref(flinitial), UCHAR_MAX);
        
    };
    
}

#endif /// LIBIMREAD_HISTOGRAM_HH_