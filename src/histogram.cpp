/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cmath>
#include <numeric>
#include <libimread/ext/valarray.hh>
#include <libimread/histogram.hh>
#include <libimread/errors.hh>
#include <libimread/image.hh>
#include <libimread/rehash.hh>

namespace im {
    
    namespace detail {
        using rehasher_t = hash::rehasher<float>;
        static const std::size_t histogram_size = UCHAR_MAX + 1;
        static const float flinitial = 0.00000000;
    }
    
    Histogram::Histogram(bytevec_t const& plane)
        // :source(plane.data(), plane.size())
        :histogram(std::cref(detail::flinitial), detail::histogram_size)
        {
            std::for_each(std::begin(plane), std::end(plane),
                          [this](byte b) { histogram[std::size_t(b)] += 1.0; });
        }
    
    Histogram::Histogram(Image const* image)
        // :source((byte*)image->rowp(), image->size())
        :histogram(std::cref(detail::flinitial), detail::histogram_size)
        {
            std::size_t imagesize = image->size();
            byte* byteptr = image->rowp_as<byte>(0);
            for (std::size_t idx = 0; idx < imagesize; ++idx) {
                histogram[std::size_t(byteptr[idx])] += 1.0;
            }
            // std::for_each(std::begin(source), std::end(source),
            //               [this](byte b) { histogram[std::size_t(b)] += 1.0; });
        }
    
    Histogram::Histogram(Image const* image, int zidx)
        :Histogram(image->template plane<byte>(zidx))
        {}
    
    Histogram::~Histogram() {}
    
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
    
    Histogram::reverse_iterator Histogram::rbegin() {
        return std::make_reverse_iterator(std::begin(histogram));
    }
    
    Histogram::reverse_iterator Histogram::rend() {
        return std::make_reverse_iterator(std::end(histogram));
    }
    
    Histogram::const_reverse_iterator Histogram::rbegin() const {
        return std::make_reverse_iterator(std::begin(histogram));
    }
    
    Histogram::const_reverse_iterator Histogram::rend() const {
        return std::make_reverse_iterator(std::end(histogram));
    }
    
    std::size_t Histogram::size() const {
        return histogram.size();
    }
    
    bool Histogram::empty() const {
        return histogram.size() == 0;
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
    
    int Histogram::min_value() const {
        /// the most infrequently occuring value
        auto result = std::find(std::begin(histogram),
                                std::end(histogram),
                                histogram.min());
        return result == std::end(histogram) ? -1 : *result;
    }
    
    int Histogram::max_value() const {
        /// the most frequently occuring value
        auto result = std::find(std::begin(histogram),
                                std::end(histogram),
                                histogram.max());
        return result == std::end(histogram) ? -1 : *result;
    }
    
    float Histogram::entropy() const {
        if (!entropy_calculated) {
            float histosize = 1.0 / histogram.sum();
            // floatva_t histofloat = histogram * histosize;
            floatva_t divisor = (histogram * histosize).apply([](float d) -> float {
                float out = d * std::log(d);
                return std::isnan(out) ? 0.00 : out;
            });
            entropy_value = -divisor.sum();
            entropy_calculated = true;
        }
        return entropy_value;
    }
    
    int Histogram::otsu() const {
        /// Otsu calculation lifted wholesale from mahotas:
        /// https://github.com/luispedro/mahotas/blob/master/mahotas/_histogram.cpp#L64-L100
        if (!otsu_calculated) {
            int n = histogram.size();
            std::vector<double> nB, nO;
            if (n <= 1) { return 0; }
            const double Hsum = std::accumulate(std::begin(histogram) + 1,
                                                std::end(histogram),
                                                double(0));
            if (Hsum == 0) { return 0; }
            
            nB.resize(n);
            nB[0] = histogram[0];
            for (int i = 1; i != n; ++i) {
                nB[i] = histogram[i] + nB[i-1];
            }
            
            nO.resize(n);
            for (int i = 0; i < n; ++i) {
                nO[i] = nB[n-1] - nB[i];
            }
            
            double mu_B = 0,
                   mu_O = 0;
            for (int i = 1; i != n; ++i) {
                mu_O += i * histogram[i];
            }
            mu_O /= Hsum;
            
            double best = nB[0] * nO[0] * (mu_B - mu_O) * (mu_B - mu_O);
            int bestT = 0;
            
            for (int T = 1; T != n; ++T) {
                if (nB[T] == 0) { continue; }
                if (nO[T] == 0) { break; }
                mu_B = (mu_B * nB[T-1] + T * histogram[T]) / nB[T];
                mu_O = (mu_O * nO[T-1] - T * histogram[T]) / nO[T];
                const double sigma_between = nB[T] * nO[T] * (mu_B - mu_O) * (mu_B - mu_O);
                if (sigma_between > best) {
                    best = sigma_between;
                    bestT = T;
                }
            }
            
            // return bestT;
            otsu_value = bestT;
            otsu_calculated = true;
        }
        return otsu_value;
    }
    
    Histogram::floatva_t Histogram::normalized() const {
        return histogram / histogram.max();
    }
    
    // Histogram::byteva_t const& Histogram::sourcedata() const {
    //     return source;
    // }
    
    Histogram::floatva_t& Histogram::values() {
        return histogram;
    }
    
    Histogram::floatva_t const& Histogram::values() const {
        return histogram;
    }
    
    float* Histogram::data() {
        return &histogram[0];
    }
    
    void Histogram::swap(Histogram& other) noexcept {
        using std::swap;
        // swap(source,             other.source);
        swap(histogram,          other.histogram);
        swap(entropy_value,      other.entropy_value);
        swap(entropy_calculated, other.entropy_calculated);
    }
    
    void swap(im::Histogram& p0, im::Histogram& p1) noexcept {
        p0.swap(p1);
    }
    
    std::size_t Histogram::hash(std::size_t seed) const noexcept {
        return std::accumulate(std::begin(histogram),
                               std::end(histogram),
                               seed, detail::rehasher_t());
    }
    
}
