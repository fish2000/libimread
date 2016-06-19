/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cmath>

#include <libimread/histogram.hh>
#include <libimread/image.hh>

namespace im {
    
    Histogram::Histogram(Image* image)
        :data(std::valarray<byte>((byte*)image->rowp(),
                                         image->size()))
        {
            std::for_each(std::begin(data), std::end(data),
                          [&histogram](byte const& b) { histogram[b] += 1.0; });
        }
    
    Histogram::begin_t Histogram::begin() {
        return std::begin(histogram);
    }
    
    Histogram::end_t Histogram::end() {
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
            std::valarray<float> histofloat = histogram * histosize;
            std::valarray<float> divisor = histofloat.apply([](float d) -> float {
                return d * std::log2(d);
            });
            entropy = -divisor.sum();
            entropy_calculated = true;
        }
        return entropy;
    }
    
    std::valarray<byte> const& Histogram::sourcedata() const {
        return data;
    }
    
    std::valarray<float>& Histogram::values() {
        return histogram;
    }
    
    std::valarray<float> const& Histogram::values() const {
        return histogram;
    }
    
}