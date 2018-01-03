/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_CROP_HH_
#define LIBIMREAD_CROP_HH_

#include <libimread/imageref.hh>
#include <libimread/ext/boundaries.hh>

namespace im {
    
    /// An im::CroppedImageRef manages an im::Image, clamped to
    /// provided maximums for its width and height dimensions.
    
    template <typename ImageType>
    class CroppedImageRef : public ImageRef<ImageType> {
        
        public:
            friend class Image;
        
        public:
            using base_t = ImageRef<ImageType>;
            using typename base_t::type;
            using typename base_t::size_type;
            using typename base_t::value_type;
            using typename base_t::reference_type;
            using typename base_t::const_reference_type;
            using typename base_t::pointer_type;
            
        public:
            using base_t::pointer;
        
        public:
            CroppedImageRef() noexcept = delete;
            CroppedImageRef(CroppedImageRef const&) = default;
            CroppedImageRef(CroppedImageRef&&) noexcept = default;
            
        public:
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass const& image, int maxX,
                                                              int maxY)
                :base_t(image)
                ,boundaries(maxX, maxY)
                {}
            
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass const& image, int minX,
                                                              int minY,
                                                              int maxX,
                                                              int maxY)
                :base_t(image)
                ,boundaries(minX, minY, maxX, maxY)
                {}
            
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass* image, int maxX,
                                                        int maxY)
                :base_t(image)
                ,boundaries(maxX, maxY)
                {}
            
            template <typename ImageClass,
                      typename std::enable_if_t<
                               std::is_base_of_v<Image, ImageClass>>>
            explicit CroppedImageRef(ImageClass* image, int minX,
                                                        int minY,
                                                        int maxX,
                                                        int maxY)
                :base_t(image)
                ,boundaries(minX, minY, maxX, maxY)
                {}
                
        public:
            explicit CroppedImageRef(const_reference_type image, int maxX,
                                                                 int maxY)
                :base_t(image)
                ,boundaries(maxX, maxY)
                {}
            
            explicit CroppedImageRef(const_reference_type image, int minX,
                                                                 int minY,
                                                                 int maxX,
                                                                 int maxY)
                :base_t(image)
                ,boundaries(minX, minY, maxX, maxY)
                {}
            
            explicit CroppedImageRef(pointer_type image_ptr, int maxX,
                                                             int maxY)
                :base_t(image_ptr)
                ,boundaries(maxX, maxY)
                {}
            
            explicit CroppedImageRef(pointer_type image_ptr, int minX,
                                                             int minY,
                                                             int maxX,
                                                             int maxY)
                :base_t(image_ptr)
                ,boundaries(minX, minY, maxX, maxY)
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
                if (Z == -1) { Z = pointer->dim_or(2); }
                /// Return a strided array view, typed accordingly,
                /// initialized with the current stride values:
                return av::strided_array_view<T, 3>(static_cast<T*>(pointer->rowp(0)),
                                                    { X, Y, Z },
                                                    { pointer->stride(0),
                                                      pointer->stride(1),
                                                      pointer->stride(2) }).section({ boundaries.width(0),
                                                                                      boundaries.height(0),
                                                                                      0 });
            }
        
        public:
            int dim_or(int dimension, int default_value = 1) const {
                switch (dimension) {
                    case 0:  return boundaries.width(pointer->dim(0));
                    case 1:  return boundaries.height(pointer->dim(1));
                    case 2:  return pointer->dim_or(2, default_value);
                    default: return default_value;
                }
            }
            
        public:
            virtual int width() const                   { return boundaries.width(pointer->dim(0)); }
            virtual int height() const                  { return boundaries.height(pointer->dim(1)); }
            virtual int size() const                    { return boundaries.width(pointer->dim(0)) *
                                                                 boundaries.height(pointer->dim(1)) *
                                                                 pointer->dim_or(2); }
            
        public:
            int left() const                            { return boundaries.width(pointer->left()); }
            int right() const                           { return boundaries.width(pointer->right()); }
            int top() const                             { return boundaries.height(pointer->top()); }
            int bottom() const                          { return boundaries.height(pointer->bottom()); }
        
        public:
            float entropy() const                       { return pointer->entropy(); }
            int otsu() const                            { return pointer->otsu(); }
        
        protected:
            boundaries_t<int> boundaries{ 0, 0, 0, 0 };
    };
    
    template <typename ImageType> inline
    CroppedImageRef<ImageType> crop(ImageType& image, int maxX, int maxY) { return CroppedImageRef<ImageType>(image, maxX, maxY); }
    
    template <typename ImageType> inline
    CroppedImageRef<ImageType> crop(ImageType& image, int minX, int minY,
                                                      int maxX, int maxY) { return CroppedImageRef<ImageType>(image, minX, minY,
                                                                                                                     maxX, maxY); }
    
} /// namespace im

#endif /// LIBIMREAD_CROP_HH_