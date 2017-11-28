/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/ext/arrayview.hh>
// #include IM_INTRINSICS_HEADER

namespace im {
    
    /// forward-declare class Histogram:
    // class Histogram;
    
    namespace detail {
        static constexpr std::size_t kHistogramSize = UCHAR_MAX + 1;
    }
    
    class Image {
        
        public:
            using size_type     = std::ptrdiff_t;
            using value_type    = byte;
            
        public:
            virtual ~Image();
            
        public:
            virtual void* rowp(int r) const = 0;
            virtual void* rowp() const;
            virtual int nbits() const = 0;
            
        public:
            virtual int nbytes() const;
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            virtual int min(int) const;
            virtual bool is_signed() const = 0;
            virtual bool is_floating_point() const = 0;
            
        public:
            template <typename T> inline
            T* rowp_as(const int r) const {
                return static_cast<T*>(rowp(r));
            }
        
        public:
            template <typename T = value_type> inline
            av::strided_array_view<T, 3> view(int X = -1,
                                              int Y = -1,
                                              int Z = -1) const {
                /// Extents default to current values:
                if (X == -1) { X = dim(0); }
                if (Y == -1) { Y = dim(1); }
                if (Z == -1) { Z = dim(2); }
                /// Return a strided array view, typed accordingly,
                /// initialized with the current stride values:
                return av::strided_array_view<T, 3>(static_cast<T*>(rowp(0)),
                                                    { X,         Y,         Z         },
                                                    { stride(0), stride(1), stride(2) });
            }
            
        public:
            int dim_or(int dimension, int default_value = 1) const;
            int stride_or(int dimension, int default_value = 1) const;
            int min_or(int dimension, int default_value = 0) const;
            
        public:
            virtual int width() const;
            virtual int height() const;
            virtual int planes() const;
            virtual int size() const;
            
        public:
            int left() const;
            int right() const;
            int top() const;
            int bottom() const;
        
        protected:
            
            class ChannelHistogram {
                
                public:
                    ChannelHistogram()
                        :internal(0.00, detail::kHistogramSize)
                        {}
                
                    float entropy();
                    int otsu();
                
                protected:
                    std::vector<double> internal;
                    bool calculated = false;
                
            };
            
        // protected:
        //     std::set<std::string> channels;
        //     std::map<std::string, Histogram> histograms;
            
        public:
            // Histogram histogram() const;
            float entropy() const;
            int otsu() const;
        
        public:
            template <typename T> inline
            std::vector<T*> allrows() const {
                using pointervec_t = std::vector<T*>;
                pointervec_t rows;
                const int h = this->dim(0);
                for (int r = 0; r != h; ++r) {
                    rows.push_back(this->rowp_as<T>(r));
                }
                return rows;
            }
            
        public:
            template <typename T, typename U = value_type> inline
            std::vector<T> plane(int idx) const {
                /// types
                using planevec_t = std::vector<T>;
                using view_t = av::strided_array_view<T, 3>;
                if (idx >= planes()) { return planevec_t{}; }
                
                /// image dimensions
                const int w = dim(0);
                const int h = dim(1);
                const int siz = w * h;
                view_t viewer = view<T>();
                
                /// TODO: Do not fill plane vector
                /// TODO: arrayview transpose?
                
                /// fill plane vector
                planevec_t out;
                out.resize(siz, 0);
                for (int x = 0; x < w; ++x) {
                    for (int y = 0; y < h; ++y) {
                        out[y * w + x] = static_cast<U>(viewer[{x, y, idx}]);
                    }
                }
                return out;
            }
            
        public:
            template <typename T, typename U = value_type> inline
            std::vector<std::vector<T>> allplanes(int lastplane = 255) const {
                using planevec_t = std::vector<T>;
                using pixvec_t = std::vector<planevec_t>;
                const int planecount = std::min(planes(), lastplane);
                pixvec_t out;
                out.reserve(planecount);
                for (int idx = 0; idx < planecount; ++idx) {
                    out.emplace_back(plane<T, U>(idx));
                }
                return out;
            }
            
    };
    
    class ImageFactory {
        
        public:
            using image_t = Image;
            using unique_t = std::unique_ptr<image_t>;
            using shared_t = std::shared_ptr<image_t>;
            
        public:
            virtual ~ImageFactory();
            
        public:
            virtual unique_t create(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
    };
    
} /* namespace im */

#endif /// LIBIMREAD_IMAGE_HH_