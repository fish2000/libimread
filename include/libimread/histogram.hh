/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HISTOGRAM_HH_
#define LIBIMREAD_HISTOGRAM_HH_

#include <vector>
#include <valarray>
#include <iterator>
#include <functional>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    /// forward-declare im::Image
    class Image;
    
    class Histogram {
        
        public:
            
            using floatva_t = std::valarray<float>;
            
            using value_type = float;
            using begin_t = decltype(std::begin(std::declval<floatva_t>()));
            using   end_t = decltype(  std::end(std::declval<floatva_t>()));
            using const_begin_t = decltype(std::begin(std::declval<floatva_t const&>()));
            using   const_end_t = decltype(  std::end(std::declval<floatva_t const&>()));
            
            using iterator = begin_t;
            using const_iterator = const_begin_t;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            
            explicit Histogram(bytevec_t const&);
            explicit Histogram(Image const*);
            explicit Histogram(Image const*, int);
            virtual ~Histogram();
            
            std::size_t size() const;
            bool empty() const;
            
            begin_t begin();
            end_t end();
            const_begin_t begin() const;
            const_end_t end() const;
            
            reverse_iterator rbegin();
            reverse_iterator rend();
            const_reverse_iterator rbegin() const;
            const_reverse_iterator rend() const;
            
            float sum() const;
            float min() const;
            float max() const;
            int min_value() const;
            int max_value() const;
            
            float entropy() const;
            int otsu() const;
            
            floatva_t normalized() const;
            floatva_t& values();
            floatva_t const& values() const;
            float* data();
            
            /// noexcept member swap
            void swap(Histogram& other) noexcept;
            friend void swap(Histogram& lhs, Histogram& rhs) noexcept;
            
            /// member hash method
            std::size_t hash(std::size_t seed = 0) const noexcept;
            
        protected:
            
            mutable float entropy_value = 0.0;
            mutable int otsu_value = 0;
            mutable bool entropy_calculated = false;
            mutable bool otsu_calculated = false;
            floatva_t histogram;
        
    };
    
}

namespace std {
    
    /// std::hash specialization for im::Histogram
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::Histogram> {
        
        typedef im::Histogram argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& histogram) const {
            return static_cast<result_type>(histogram.hash());
        }
        
    };
    
}; /* namespace std */

#endif /// LIBIMREAD_HISTOGRAM_HH_