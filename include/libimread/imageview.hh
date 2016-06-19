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
    
    class ImageView {
        
        public:
            // friend class Image;
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
            
            ImageView(ImageView const& other)
                :source(other.source)
                {}
                
            ImageView(ImageView&& other) noexcept
                :source(std::move(other.source))
                {}
            
            // template <typename ImageType>
            // explicit ImageView(ImageType const& image)
            //     :source(new ImageType(image))
            //     {}
            
            // explicit ImageView(Image const& image)
            //     :ImageView(new ImageView(image, typename Tag::Heap{}), typename Tag::Heap{})
            //     {}
            //
            // explicit ImageView(Image const& image, typename Tag::Heap)
            //     :source(image)
            //     {}
            //
            explicit ImageView(Image* image)
                :source(image)
                {}
            
            explicit ImageView(shared_image_t shared_image);
            explicit ImageView(const_shared_image_t shared_image);
            explicit ImageView(const_unique_image_t unique_image, typename Tag::Unique);
            
            ImageView& operator=(ImageView const& other) {
                ImageView(other).swap(*this);
                return *this;
            }
            
            ImageView& operator=(ImageView&& other) noexcept {
                ImageView(std::move(other)).swap(*this);
                return *this;
            }
            
            /// take ownership when assigned with an image pointer
            ImageView& operator=(Image* image_ptr) {
                source = const_shared_image_t(image_ptr);
                return *this;
            }
            
            /// delegate the core Image API methods
            /// back to the source image
            virtual void* rowp(int r) const;
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
            
            virtual shared_image_t shared();
            virtual weak_image_t weak();
            
            void swap(ImageView& other) {
                using std::swap;
                swap(source, other.source);
            }
            friend void swap(ImageView& lhs, ImageView& rhs) {
                using std::swap;
                swap(lhs.source, rhs.source);
            }
            
        protected:
            const_shared_image_t source;
            ImageView(void);
    };
    
}

#endif /// LIBIMREAD_IMAGEVIEW_HH_