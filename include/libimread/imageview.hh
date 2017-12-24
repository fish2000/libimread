/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEVIEW_HH_
#define LIBIMREAD_IMAGEVIEW_HH_

#include <vector>
#include <memory>
#include <utility>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/ext/arrayview.hh>


namespace im {
    
    /// forward-declare im::Image and im::Histogram
    class Image;
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
            
            ImageView& operator=(ImageView const& other) &;
            ImageView& operator=(ImageView&& other) & noexcept;
            ImageView& operator=(Image* image_ptr) &;
            
            /// delegate the core Image API methods
            /// back to the source image
            virtual void* rowp(int r) const;
            virtual void* rowp() const;
            virtual int nbits() const;
            virtual int nbytes() const;
            virtual int ndims() const;
            virtual int dim(int d) const;
            virtual int stride(int s) const;
            virtual int min(int s) const;
            virtual bool is_signed() const;
            virtual bool is_floating_point() const;
            
            virtual int dim_or(int dim, int default_value = 1) const;
            virtual int stride_or(int dim, int default_value = 1) const;
            virtual int min_or(int dim, int default_value = 0) const;
            
            virtual int width() const;
            virtual int height() const;
            virtual int planes() const;
            virtual int size() const;
            virtual int left() const;
            virtual int right() const;
            virtual int top() const;
            virtual int bottom() const;
            
            virtual Histogram histogram() const;
            virtual float entropy() const;
            
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
            
            template <typename T> inline
            std::vector<T*> allrows() const {
                using pointervec_t = std::vector<T*>;
                pointervec_t rows;
                const int h = dim(0);
                for (int r = 0; r != h; ++r) {
                    rows.push_back(rowp_as<T>(r));
                }
                return rows;
            }
            
            virtual shared_imageview_t shared();
            virtual weak_imageview_t weak();
            
            std::size_t hash(std::size_t seed = 0) const noexcept;
            void swap(ImageView& other);
            friend void swap(ImageView& lhs, ImageView& rhs);
            
        protected:
            Image* source;
            
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