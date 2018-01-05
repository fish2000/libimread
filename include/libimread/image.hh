/// Copyright 2012-2018 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/accessors.hh>
#include <libimread/metadata.hh>
#include <libimread/histogram.hh>
#include <libimread/imageref.hh>
// #include IM_INTRINSICS_HEADER

namespace im {
    
    namespace detail {
        static constexpr std::size_t kHistogramSize = UCHAR_MAX + 1;
    }
    
    class ImageView;
    
    class Image {
        
        public:
            using size_type     = std::ptrdiff_t;
            using value_type    = byte;
            friend class ImageView;
        
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
            /// Accessor definition macros -- q.v. accessors.hh:
            IMAGE_ACCESSOR_ROWP_AS(this);
            IMAGE_ACCESSOR_VIEW(this);
            IMAGE_ACCESSOR_ALLROWS(this);
            IMAGE_ACCESSOR_PLANE(this);
            IMAGE_ACCESSOR_ALLPLANES(this);
        
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
        
        // protected:
        //
        //     class ChannelHistogram {
        //
        //         public:
        //             ChannelHistogram()
        //                 :internal(0.00, detail::kHistogramSize)
        //                 {}
        //
        //             float entropy();
        //             int otsu();
        //
        //         protected:
        //             std::vector<double> internal;
        //             bool calculated = false;
        //
        //     };
        
        // protected:
        //     std::set<std::string> channels;
        //     std::map<std::string, Histogram> histograms;
        
        public:
            float entropy() const;
            int otsu() const;
        
        public:
            Metadata&       metadata();
            Metadata const& metadata() const;
            Metadata&       metadata(Metadata&);
            Metadata&       metadata(Metadata&&);
            Metadata*       metadata_ptr();
            Metadata const* metadata_ptr() const;
        
        protected:
            Metadata md;
            mutable std::shared_ptr<Histogram> histo;
        
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