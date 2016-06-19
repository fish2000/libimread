/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/pixels.hh>

namespace im {
    
    class Image : public std::enable_shared_from_this<Image> {
        
        public:
            using unique_image_t = std::unique_ptr<Image>;
            using shared_image_t = std::shared_ptr<Image>;
            using weak_image_t   = std::weak_ptr<Image>;
            using image_ptr_t    = std::add_pointer_t<Image>;
            
            using const_unique_image_t = std::unique_ptr<Image const>;
            using const_shared_image_t = std::shared_ptr<Image const>;
            using const_weak_image_t   = std::weak_ptr<Image const>;
            using const_image_ptr_t    = std::add_pointer_t<Image const>;
            
            // enum class Type : std::size_t {
            //     UINT8   = sizeof(uint8_t),
            //     INT32   = sizeof(int32_t),
            //     FLOAT   = 128+sizeof(float),
            //     DOUBLE  = 128+sizeof(double),
            //     VOIDPTR = 256+sizeof(std::ptrdiff_t)
            // };
            
            virtual ~Image() {}
            
            virtual void* rowp(int r) const = 0;
            virtual int nbits() const = 0;
            
            virtual int nbytes() const;
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            virtual bool is_signed() const = 0;
            virtual bool is_floating_point() const = 0;
            
            virtual int dim_or(int dim, int default_value = 1) const;
            virtual int stride_or(int dim, int default_value = 1) const;
            virtual int width() const;
            virtual int height() const;
            virtual int planes() const;
            virtual int size() const;
            
            virtual shared_image_t shared();
            virtual const_shared_image_t shared() const;
            virtual weak_image_t weak();
            virtual const_weak_image_t weak() const;
            
            template <typename T> inline
            T* rowp_as(const int r) const {
                return static_cast<T*>(this->rowp(r));
            }
            
            template <typename T = byte> inline
            pix::accessor<T> access() const {
                return pix::accessor<T>(rowp_as<T>(0), stride(0),
                                                       stride(1),
                                                       stride(2));
            }
            
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
            
            template <typename T, typename U = byte> inline
            std::vector<T> plane(int idx) const {
                /// types
                using planevec_t = std::vector<T>;
                using access_t = pix::accessor<T>;
                if (idx >= planes()) { return planevec_t{}; }
                
                /// image dimensions
                const int w = dim(0);
                const int h = dim(1);
                const int siz = w * h;
                access_t accessor = access<T>();
                
                /// fill plane vector
                planevec_t out(siz);
                out.resize(siz);
                for (int x = 0; x < w; x++) {
                    for (int y = 0; y < h; y++) {
                        pix::convert(static_cast<U>(
                            accessor(x, y, idx)[0]),
                                     out[y * w + x]);
                    }
                }
                return out;
            }
            
            template <typename T, typename U = byte> inline
            std::vector<std::vector<T>> allplanes(int lastplane = 255) const {
                using planevec_t = std::vector<T>;
                using pixvec_t = std::vector<planevec_t>;
                const int planecount = std::min(planes(), lastplane);
                pixvec_t out(planecount);
                for (int idx = 0; idx < planecount; idx++) {
                    out.push_back(std::move(plane<T, U>(idx)));
                }
                return out;
            }
            
    };
    
    class ImageFactory {
        
        public:
            using image_t = Image;
            using unique_t = std::unique_ptr<image_t>;
            using shared_t = std::shared_ptr<image_t>;
            
            virtual ~ImageFactory();
            
            virtual unique_t create(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
            
            virtual shared_t shared(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
            
        protected:
            template <typename T>
            std::shared_ptr<T> specialized(int nbits,
                           int d0, int d1, int d2,
                           int d3=-1, int d4=-1) {
                return std::dynamic_pointer_cast<T>(
                    shared(nbits, d0, d1, d2, d3, d4));
            };
    };
    
    class ImageWithMetadata {
        public:
            ImageWithMetadata();
            ImageWithMetadata(std::string const& m);
            virtual ~ImageWithMetadata();
            
            std::string const& get_meta() const;
            void set_meta(std::string const& m);
            
        private:
            std::string meta;
    };
    
    /// This class *owns* its members and will delete them if destroyed
    struct ImageList {
        using pointer_t      = std::add_pointer_t<Image>;
        using unique_t       = std::unique_ptr<Image>;
        using vector_t       = std::vector<pointer_t>;
        using pointerlist_t  = std::initializer_list<pointer_t>;
        using vector_size_t  = vector_t::size_type;
        using iterator       = vector_t::iterator;
        using const_iterator = vector_t::const_iterator;
        
        /// default constructor
        ImageList() noexcept = default;
        
        /// construct from multiple arguments
        /// ... using boolean tag for first arg
        template <typename ...Pointers>
        explicit ImageList(bool pointerargs, Pointers ...pointers)
            :content{ pointers... }
            {
                content.erase(
                    std::remove_if(content.begin(), content.end(),
                                [](pointer_t p) { return p == nullptr; }),
                    content.end());
            }
        
        /// initializer list construction
        explicit ImageList(pointerlist_t pointerlist);
        
        /// move-construct from pointer vector
        explicit ImageList(vector_t&& vector);
        
        /// noexcept move constructor
        ImageList(ImageList&& other) noexcept;
        
        /// noexcept move assignment operator
        ImageList& operator=(ImageList&& other) noexcept;
        
        vector_size_t size() const;
        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        
        void erase(iterator it);
        void prepend(pointer_t image);
        void push_front(pointer_t image);
        void append(pointer_t image);
        void push_back(pointer_t image);
        void push_back(unique_t unique);
        
        pointer_t get(vector_size_t idx) const;
        pointer_t at(vector_size_t idx) const;
        unique_t yank(vector_size_t idx);
        unique_t pop();
        void reset();
        void reset(vector_t&& vector);
        ~ImageList();
        
        /// After calling release(), ownership of the content image ponters
        /// is transferred to the caller, who must figure out how to delete them.
        /// Also note that release() resets the internal vector.
        vector_t release();
        
        /// noexcept member swap
        void swap(ImageList& other) noexcept;
        
        /// member hash method
        std::size_t hash(std::size_t seed = 0) const noexcept;
        
        private:
            ImageList(const ImageList&);
            ImageList &operator=(const ImageList&);
            vector_t content;
    };
    
} /* namespace im */

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept;
    
    /// std::hash specialization for filesystem::path
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::ImageList> {
        
        typedef im::ImageList argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& list) const {
            return static_cast<result_type>(list.hash());
        }
        
    };
    
}; /* namespace std */


#endif /// LIBIMREAD_IMAGE_HH_