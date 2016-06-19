/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cmath>
#include <libimread/ext/valarray.hh>
#include <libimread/histogram.hh>
#include <libimread/image.hh>

namespace im {
    
    Histogram::Histogram(Image* image)
        :data(byteva_t((byte*)image->rowp(),
                              image->size()))
        ,histogram(std::cref(flinitial), UCHAR_MAX)
        {
            std::for_each(std::begin(data), std::end(data),
                          [this](byte const& b) { histogram[b] += 1.0; });
        }
    
    Histogram::begin_t Histogram::begin() {
        return std::begin(histogram);
    }
    
    Histogram::end_t Histogram::end() {
        return std::end(histogram);
    }
    
    Histogram::const_begin_t Histogram::begin() const {
        return std::begin(histogram);
    }
    
    Histogram::const_end_t Histogram::end() const {
        return std::end(histogram);
    }
    
    std::size_t Histogram::size() const {
        return histogram.size();
    }
    
    float Histogram::sum() const {
        return histogram.sum();
    }
    
    float Histogram::min() const {
        return histogram.min();
    }
    
    float Histogram::max() const {
        return histogram.max();
    }
    
    float Histogram::entropy() const {
        if (!entropy_calculated) {
            float histosize = 1.0 / float(histogram.sum());
            floatva_t histofloat = histogram * histosize;
            floatva_t divisor = histofloat.apply([](float d) -> float {
                return d * std::log2(d);
            });
            entropy_value = -divisor.sum();
            entropy_calculated = true;
        }
        return entropy_value;
    }
    
    Histogram::floatva_t Histogram::normalized() const {
        floatva_t out = histogram / histogram.max();
        return out;
    }
    
    Histogram::byteva_t const& Histogram::sourcedata() const {
        return data;
    }
    
    Histogram::floatva_t& Histogram::values() {
        return histogram;
    }
    
    Histogram::floatva_t const& Histogram::values() const {
        return histogram;
    }
    
}