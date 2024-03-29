/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEVIEW_HH_
#define LIBIMREAD_IMAGEVIEW_HH_

#include <memory>
#include <utility>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/accessors.hh>
#include <libimread/metadata.hh>

namespace im {
    
    /// forward-declare im::Histogram
    class Histogram;
    
    class ImageView : public std::enable_shared_from_this<ImageView> {
        
        public:
            using size_type     = std::ptrdiff_t;
            using value_type    = byte;
        
        public:
            using image_ptr_t    = std::add_pointer_t<Image>;
            
        public:
            using shared_imageview_t = std::shared_ptr<ImageView>;
            using weak_imageview_t   = std::weak_ptr<ImageView>;
            using imageview_ptr_t    = std::add_pointer_t<ImageView>;
            
        public:
            ImageView(ImageView const& other);
            ImageView(ImageView&& other) noexcept;
            explicit ImageView(Image* image);
            virtual ~ImageView();
            
        public:
            ImageView& operator=(ImageView const& other) &;
            ImageView& operator=(ImageView&& other) & noexcept;
            ImageView& operator=(Image* image_ptr) &;
            
        public:
            /// delegate the core Image API methods
            /// back to the source image
            virtual void* rowp(int r) const;
            virtual void* rowp() const;
            
        public:
            virtual int nbits() const;
            virtual int nbytes() const;
            virtual int ndims() const;
            virtual int dim(int d) const;
            virtual int stride(int s) const;
            virtual int min(int s) const;
            virtual bool is_signed() const;
            virtual bool is_floating_point() const;
            
        public:
            int dim_or(int dim, int default_value = 1) const;
            int stride_or(int dim, int default_value = 1) const;
            int min_or(int dim, int default_value = 0) const;
            
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
            
        public:
            // Histogram histogram() const;
            float entropy() const;
            int otsu() const;
        
        public:
            Metadata&       metadata();
            Metadata const& metadata() const;
            Metadata&       metadata(Metadata&);
            Metadata&       metadata(Metadata&&);
            Metadata*       metadata_ptr();
            Metadata const* metadata_ptr() const;
            
        public:
            /// Accessor definition macros -- q.v. accessors.hh:
            IMAGE_ACCESSOR_ROWP_AS(source);
            IMAGE_ACCESSOR_VIEW(source);
            IMAGE_ACCESSOR_ALLROWS(source);
            IMAGE_ACCESSOR_PLANE(source);
            IMAGE_ACCESSOR_ALLPLANES(source);
            
        public:
            virtual shared_imageview_t shared();
            virtual weak_imageview_t weak();
            
        public:
            std::size_t hash(std::size_t seed = 0) const noexcept;
            void swap(ImageView& other);
            friend void swap(ImageView& lhs, ImageView& rhs);
            
        protected:
            Image* source;
            mutable std::unique_ptr<Histogram> histo;
            
        private:
            ImageView(void);
    };
    
}

namespace std {
    
    /// std::hash specialization for im::ImageView
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::ImageView> {
        
        typedef im::ImageView argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& imageview) const {
            return static_cast<result_type>(imageview.hash());
        }
        
    };
    
}; /* namespace std */

#endif /// LIBIMREAD_IMAGEVIEW_HH_