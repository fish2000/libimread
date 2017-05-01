
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <iostream>
#include <valarray>
#include <iterator>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <iomanip>
#include <random>

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


using byte = uint8_t;

template <typename DataType>
using randomizer_t = std::independent_bits_engine<
                     std::default_random_engine,
                     sizeof(DataType)*8, DataType>;

template <typename ValueType>
std::ostream& operator<<(std::ostream& out,
                         std::valarray<ValueType> const& varray) {
    if (std::is_floating_point<ValueType>::value) {
        out << std::dec << std::fixed
            << std::setprecision(2)
            << std::setw(2)
            << std::showpoint;
    } else {
        out << std::fixed
            << std::noshowpoint;
    }
    
    std::for_each(std::begin(varray),
                  std::end(varray),
           [&out](ValueType const& value) {
        out << std::to_string(value)
            << ", ";
    });
    
    return out;
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


namespace {
    extern "C" {
        int main(void) {
            
            byte initial = 0x00;
            float flinitial = 0.00000000;
            randomizer_t<byte> randomizer;
            std::vector<byte> source(320*240*1, std::cref(initial));
            std::generate(std::begin(source),
                          std::end(source),
                          std::ref(randomizer));
            
            // std::valarray<byte> histogram(source.data(), source.size());
            
            __attribute__((unused))
            std::array<float, 255> fixed{{ flinitial }};
            
            std::valarray<float> histogram(std::cref(flinitial), 255);
            std::for_each(source.begin(), source.end(),
                          [&](byte const& b) { histogram[b] += 1.0; });
            
            // __attribute__((unused))
            
            float histosize = 1.0 / float(histogram.sum());
            std::valarray<float> histofloat = histogram * histosize;
            std::valarray<float> divisor = histofloat.apply([](float d) -> float {
                return d * std::log2(d);
            });
            float entropy = -divisor.sum();
            
            // std::valarray<byte> wat = divisor.apply([](float d) -> float {
            //     return detail::clamp(int(d), 0, 255);
            // });
            // std::valarray<float> adjusted = histogram - 255.0;
            // std::valarray<byte> wat = valarray::cast<byte>(histogram, [](float d) -> byte {
            //     return detail::clamp(int(d - 255.0), 0, 255);
            // });
            // std::valarray<byte> wat = valarray::cast<byte>(source);
            std::valarray<byte> wat = valarray::cast<byte>(histogram);
            // std::valarray<byte> wat = valarray::cast<byte>(fixed); /// BZZT: needs different template-template params
            
            std::cout << "YO DOGG: " << std::endl;
            std::cout << "histogram.size(): " << histogram.size() << std::endl;
            std::cout << "entropy: " << entropy << std::endl;
            std::cout << "histogram: " << histogram << std::endl;
            std::cout << "wat: " << wat << std::endl;
            
            std::cout << std::endl << std::endl << std::endl;
            std::cout << "sizeof(unsigned short): " << sizeof(unsigned short) << std::endl;
            
            return 0;
        }
    }
}


