/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_CROP_HH_
#define LIBIMREAD_CROP_HH_

#include <algorithm>
#include <functional>
#include <libimread/imageref.hh>

namespace im {
    
    namespace detail {
        
        struct ARG {} arg;
        
    }
    
}

namespace std {
    
    template <>
    struct is_placeholder<im::detail::ARG> : public std::integral_constant<int, 1> {};
    
}

namespace im {
    
    template <typename T>
    using clamper_f = std::function<T&(T&...)>;
    
    template <typename T,
              typename = std::enable_if_t<std::is_arithmetic<
                                          std::remove_cv_t<T>>::value>
    clamper_f<T> clamper(T& lower_bound, T& upper_bound) {
        return std::bind(std::clamp<T>, detail::arg, lower_bound, upper_bound);
    }
    
    template <typename T,
              typename = std::enable_if_t<std::is_arithmetic<
                                          std::remove_cv_t<T>>::value>
    clamper_f<T> clamper(T& upper_bound) {
        return std::bind(std::clamp<T>, detail::arg, 0, upper_bound);
    }
    
    /// An im::CroppedImageRef manages an im::Image, clamped to
    /// provided maximums for its width and height dimensions.
    
    template <typename ImageType>
    class CroppedImageRef : public ImageRef<ImageType> {
        
        public:
            friend class Image;
        
        public:
            using base = ImageRef<ImageType>;
            using base::type;
            using base::size_type;
            using base::value_type;
            using base::reference_type;
            using base::const_reference_type;
            using base::pointer_type;
            
        public:
            using base::pointer;
        
        public:
            CroppedImageRef() noexcept = delete;
            CroppedImageRef(CroppedImageRef const&) = default;
            CroppedImageRef(CroppedImageRef&&) noexcept = default;
            
        public:
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass const& image, int crop_to_width,
                                                              int crop_to_height)
                :ImageRef(image)
                ,boundaries(crop_to_width,
                            crop_to_height)
                {}
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass* image, int crop_to_width,
                                                        int crop_to_height)
                :ImageRef(image)
                ,boundaries(crop_to_width,
                            crop_to_height)
                {}
                
        public:
            explicit CroppedImageRef(const_reference_type image, int crop_to_width,
                                                                 int crop_to_height)
                :ImageRef(image)
                ,boundaries(crop_to_width,
                            crop_to_height)
                {}
            explicit CroppedImageRef(pointer_type image_ptr, int crop_to_width,
                                                             int crop_to_height)
                :ImageRef(image_ptr)
                ,boundaries(crop_to_width,
                            crop_to_height)
                {}
        
        public:
            virtual ~CroppedImageRef() {}
        
        public:
            virtual void* rowp(int r) const             { return pointer->rowp(boundaries.height(r)); }
            
        public:
            virtual int dim(int d) const {
                switch (d) {
                    case 0:  return boundaries.width(pointer->dim(0));
                    case 1:  return boundaries.height(pointer->dim(1));
                    default: return pointer->dim(d);
                }
            }
            
            virtual int stride(int s) const {
                switch (s) {
                    case 0:  return boundaries.width(pointer->stride(0));
                    case 1:  return boundaries.height(pointer->stride(1));
                    default: return pointer->stride(s);
                }
            }
            
            virtual int min(int m) const {
                switch (m) {
                    case 0:  return boundaries.width(pointer->min(0));
                    case 1:  return boundaries.height(pointer->min(1));
                    default: return pointer->min(m);
                }
            }
        
        public:
            template <typename T> inline
            T* rowp_as(const int r) const               { return static_cast<T*>(pointer->rowp(boundaries.height(r))); }
        
        public:
            template <typename T = value_type> inline
            av::strided_array_view<T, 3> view(int X = -1,
                                              int Y = -1,
                                              int Z = -1) const {
                /// Extents default to current values:
                if (X == -1) { X = boundaries.width(pointer->dim(0)); }
                if (Y == -1) { Y = boundaries.height(pointer->dim(1)); }
                if (Z == -1) { Z = pointer->dim(2); }
                /// Return a strided array view, typed accordingly,
                /// initialized with the current stride values:
                return av::strided_array_view<T, 3>(static_cast<T*>(pointer->rowp()),
                                                    { X, Y, Z },
                                                    { boundaries.width(pointer->stride(0)),
                                                      boundaries.height(pointer->stride(1)),
                                                      pointer->stride(2) });
            }
        
        public:
            int dim_or(int dimension, int default_value = 1) const {
                switch (dimension) {
                    case 0:  return boundaries.width(pointer->dim(0));
                    case 1:  return boundaries.height(pointer->dim(1));
                    default: return pointer->dim_or(dimension, default_value);
                }
            }
            
            int stride_or(int dimension, int default_value = 1) const  {
                switch (dimension) {
                    case 0:  return boundaries.width(pointer->stride(0));
                    case 1:  return boundaries.height(pointer->stride(1));
                    default: return pointer->stride_or(dimension, default_value);
                }
            }
            
            int min_or(int dimension, int default_value = 0) const {
                switch (dimension) {
                    case 0:  return boundaries.width(pointer->min(0));
                    case 1:  return boundaries.height(pointer->min(1));
                    default: return pointer->min_or(dimension, default_value);
                }
            }
            
        public:
            virtual int width() const                   { return boundaries.width(pointer->width()); }
            virtual int height() const                  { return boundaries.height(pointer->height()); }
            virtual int planes() const                  { return pointer->planes(); }
            virtual int size() const                    { return boundaries.width(pointer->width()) *
                                                                 boundaries.height(pointer->height()) *
                                                                 pointer->planes(); }
            
        public:
            int left() const                            { return boundaries.width(pointer->left()); }
            int right() const                           { return boundaries.width(pointer->right()); }
            int top() const                             { return boundaries.height(pointer->top()); }
            int bottom() const                          { return boundaries.height(pointer->bottom()); }
        
        public:
            float entropy() const                       { return pointer->entropy(); }
            int otsu() const                            { return pointer->otsu(); }
        
        protected:
            struct boundaries_t {
                
                clamper_f<int> width;
                clamper_f<int> height;
                
                boundaries_t(int w, int h)
                    :width(clamper(w))
                    ,height(clamper(h))
                    {}
                
            } boundaries;
        
    };
    
} /// namespace im

#endif /// LIBIMREAD_CROP_HH_