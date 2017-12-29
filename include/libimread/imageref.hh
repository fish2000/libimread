/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEREF_HH_
#define LIBIMREAD_IMAGEREF_HH_

#include <vector>
#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/metadata.hh>
#include <libimread/ext/arrayview.hh>

namespace im {
    
    /// We define im::ImageRef<…> before im::Image:
    class Image;
    
    template <typename ImageType>
    class ImageRef {
        
        public:
            friend class Image;
        
        public:
            using type = typename std::remove_cv_t<std::decay_t<ImageType>>;
            using size_type = typename type::size_type;
            using value_type = typename type::value_type;
            using reference_type = typename std::add_lvalue_reference_t<type>;
            using const_reference_type = typename std::add_const_t<reference_type>;
            using pointer_type = typename std::add_pointer_t<type>;
        
        public:
            ImageRef() noexcept = delete;
            ImageRef(ImageRef const&) = default;
            ImageRef(ImageRef&&) noexcept = default;
            
        public:
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit ImageRef(ImageClass const& image)
                :pointer(std::addressof(image))
                {}
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit ImageRef(ImageClass* image)
                :pointer(image)
                { /* ¿NON-NULL? */ }
                
        public:
            explicit ImageRef(const_reference_type image)
                :pointer(std::addressof(image))
                {}
            explicit ImageRef(pointer_type image_ptr)
                :pointer(image_ptr)
                { /* ¿NON-NULL? */ }
        
        public:
            bool is_valid() const { return pointer != nullptr; }
        
        public:
            virtual ~ImageRef() {}
        
        public:
            virtual void* rowp(int r) const { return pointer->rowp(r); }
            virtual void* rowp() const { return pointer->rowp(); }
            virtual int nbits() const { return pointer->nbits(); }
            
        public:
            virtual int nbytes() const { return pointer->nbytes(); }
            virtual int ndims() const { return pointer->ndims(); }
            virtual int dim(int d) const { return pointer->dim(d); }
            virtual int stride(int s) const { return pointer->stride(s); }
            virtual int min(int m) const { return pointer->min(m); }
            virtual bool is_signed() const { return pointer->is_signed(); }
            virtual bool is_floating_point() const { return pointer->is_floating_point(); }
        
        public:
            template <typename T> inline
            T* rowp_as(const int r) const {
                return static_cast<T*>(pointer->rowp(r));
            }
        
        public:
            template <typename T = value_type> inline
            av::strided_array_view<T, 3> view(int X = -1,
                                              int Y = -1,
                                              int Z = -1) const {
                /// Extents default to current values:
                if (X == -1) { X = pointer->dim(0); }
                if (Y == -1) { Y = pointer->dim(1); }
                if (Z == -1) { Z = pointer->dim(2); }
                /// Return a strided array view, typed accordingly,
                /// initialized with the current stride values:
                return av::strided_array_view<T, 3>(static_cast<T*>(pointer->rowp(0)),
                                                    { X,                  Y,                  Z                  },
                                                    { pointer->stride(0), pointer->stride(1), pointer->stride(2) });
            }
            
        public:
            int dim_or(int dimension, int default_value = 1) const { return pointer->dim_or(dimension, default_value); }
            int stride_or(int dimension, int default_value = 1) const { return pointer->stride_or(dimension, default_value); }
            int min_or(int dimension, int default_value = 0) const { return pointer->min_or(dimension, default_value); }
            
        public:
            virtual int width() const { return pointer->width(); }
            virtual int height() const { return pointer->height(); }
            virtual int planes() const { return pointer->planes(); }
            virtual int size() const { return pointer->size(); }
            
        public:
            int left() const { return pointer->left(); }
            int right() const { return pointer->right(); }
            int top() const { return pointer->top(); }
            int bottom() const { return pointer->bottom(); }
        
        public:
            float entropy() const { return pointer->entropy(); }
            int otsu() const { return pointer->otsu(); }
        
        public:
            Metadata&       metadata()                  { return pointer->metadata();                               }
            Metadata const& metadata() const            { return pointer->metadata();                               }
            Metadata&       metadata(Metadata& m)       { return pointer->metadata(m);                              }
            Metadata&       metadata(Metadata&& m)      { return pointer->metadata(std::forward<Metadata&&>(m));    }
            Metadata*       metadata_ptr()              { return pointer->metadata_ptr();                           }
            Metadata const* metadata_ptr() const        { return pointer->metadata_ptr();                           }
        
        protected:
            pointer_type pointer{ nullptr };
        
    };
    
} /// namespace im

#endif /// LIBIMREAD_IMAGEREF_HH_