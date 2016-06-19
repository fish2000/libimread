/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HISTOGRAM_HH_
#define LIBIMREAD_HISTOGRAM_HH_

#include <valarray>
#include <iterator>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <utility>

#include <libimread/libimread.hpp>

namespace im {
    
    /// forward-declare im::Image
    class Image;
    
    class Histogram {
        
        using begin_t = decltype(std::begin(std::declval<std::valarray<float>>()));
        using   end_t = decltype(  std::end(std::declval<std::valarray<float>>()));
        
        public:
            explicit Histogram(Image*);
            
            std::size_t size() const;
            begin_t begin();
            end_t end();
            float sum() const;
            float min() const;
            float max() const;
            float entropy();
            std::valarray<byte> const& sourcedata() const;
            std::valarray<float>& values();
            std::valarray<float> const& values() const;
        
        protected:
            float flinitial = 0.00000000;
            float entropy_value = 0.0;
            bool entropy_calculated = false;
            std::valarray<byte> data;
            std::valarray<float> histogram;
        
    };
    
}

#endif /// LIBIMREAD_HISTOGRAM_HH_