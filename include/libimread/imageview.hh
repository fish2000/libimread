/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEVIEW_HH_
#define LIBIMREAD_IMAGEVIEW_HH_

#include <memory>
#include <utility>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    class Image;
    
    class ImageView : public std::enable_shared_from_this<ImageView> {
        
        public:
            struct Tag {
                struct Shared   {};
                struct Unique   {};
            };
            
            using unique_image_t = std::unique_ptr<Image>;
            using shared_image_t = std::shared_ptr<Image>;
            using weak_image_t   = std::weak_ptr<Image>;
            using image_ptr_t    = std::add_pointer_t<Image>;
            
            using const_unique_image_t = std::unique_ptr<Image const>;
            using const_shared_image_t = std::shared_ptr<Image const>;
            using const_weak_image_t   = std::weak_ptr<Image const>;
            using const_image_ptr_t    = std::add_pointer_t<Image const>;
            
            using shared_imageview_t = std::shared_ptr<ImageView>;
            using weak_imageview_t   = std::weak_ptr<ImageView>;
            using imageview_ptr_t    = std::add_pointer_t<ImageView>;
            
            ImageView(ImageView const& other);
            ImageView(ImageView&& other) noexcept;
            explicit ImageView(Image* image);
            
            ImageView& operator=(ImageView const& other);
            ImageView& operator=(ImageView&& other) noexcept;
            ImageView& operator=(Image* image_ptr);
            
            /// delegate the core Image API methods
            /// back to the source image
            virtual void* rowp(int r) const;
            virtual void* rowp() const;
            virtual int nbits() const;
            virtual int nbytes() const;
            virtual int ndims() const;
            virtual int dim(int d) const;
            virtual int stride(int s) const;
            virtual bool is_signed() const;
            virtual bool is_floating_point() const;
            
            virtual int dim_or(int dim, int default_value = 1) const;
            virtual int stride_or(int dim, int default_value = 1) const;
            virtual int width() const;
            virtual int height() const;
            virtual int planes() const;
            virtual int size() const;
            
            virtual shared_imageview_t shared();
            virtual weak_imageview_t weak();
            
            void swap(ImageView& other);
            friend void swap(ImageView& lhs, ImageView& rhs);
            
        protected:
            Image* source;
            ImageView(void);
    };
    
}

#endif /// LIBIMREAD_IMAGEVIEW_HH_