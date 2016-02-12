/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/pixels.hh>

namespace im {
    
    class Image {
        
        public:
            virtual ~Image() {}
            
            virtual void* rowp(int r) const = 0;
            virtual int nbits() const = 0;
            
            virtual int nbytes() const {
                const int bits = this->nbits();
                return (bits / 8) + bool(bits % 8);
            }
            
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            
            virtual int dim_or(int dim, int default_value = 1) const {
                if (dim >= this->ndims()) { return default_value; }
                return this->dim(dim);
            }
            
            virtual int stride_or(int dim, int default_value = 1) const {
                if (dim >= this->ndims()) { return default_value; }
                return this->stride(dim);
            }
            
            virtual int planes() const { return this->dim_or(2); }
            virtual int size() const {
                return dim_or(0) * dim_or(1) * dim_or(2) * dim_or(3);
            }
            
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
            
            virtual ~ImageFactory() { }
            
            virtual std::unique_ptr<image_t>
                create(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
            
            virtual std::shared_ptr<image_t>
                shared(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
            
        protected:
            template <typename T>
            std::shared_ptr<T>
                specialized(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) {
                        return std::dynamic_pointer_cast<T>(
                            shared(nbits, d0, d1, d2, d3, d4));
                    };
    };
    
    class ImageWithMetadata {
        public:
            ImageWithMetadata()
                :meta("")
                {}
            ImageWithMetadata(const std::string& m)
                :meta(m)
                {}
            
            virtual ~ImageWithMetadata() {}
            
            const std::string& get_meta() const { return meta; }
            void set_meta(const std::string& m) { meta = m; }
            
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
        
        /// initializer list construction
        explicit ImageList(pointerlist_t pointerlist)
            :content(pointerlist)
            {
                content.erase(
                    std::remove_if(content.begin(), content.end(),
                                [](pointer_t p) { return p == nullptr; }),
                    content.end());
            }
        
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
        
        /// move-construct from pointer vector
        explicit ImageList(vector_t&& vector)
            :content(std::move(vector))
            {
                content.erase(
                    std::remove_if(content.begin(), content.end(),
                                [](pointer_t p) { return p == nullptr; }),
                    content.end());
            }
        
        /// noexcept move constructor
        ImageList(ImageList&& other) noexcept
            :content(std::move(other.release()))
            {}
        
        /// noexcept move assignment operator
        ImageList& operator=(ImageList&& other) noexcept {
            if (content != other.content) {
                content = std::move(other.release());
            }
            return *this;
        }
        
        vector_size_t size() const          { return content.size(); }
        iterator begin()                    { return content.begin(); }
        iterator end()                      { return content.end(); }
        const_iterator begin() const        { return content.begin(); }
        const_iterator end()   const        { return content.end(); }
        
        void erase(iterator it)             { content.erase(it); }
        void prepend(pointer_t image)       { content.insert(content.begin(), image); }
        void push_front(pointer_t image)    { content.insert(content.begin(), image); }
        void append(pointer_t image)        { content.push_back(image); }
        void push_back(pointer_t image)     { content.push_back(image); }
        
        void push_back(unique_t unique) {
            content.push_back(unique.release());
        }
        
        pointer_t get(vector_size_t idx) const   { return content[idx]; }
        pointer_t at(vector_size_t idx) const    { return content.at(idx); }
        
        unique_t yank(vector_size_t idx) {
            /// remove the pointer at idx, resizing the internal vector;
            /// return it as managed by a new unique_ptr
            /// ... this'll throw std::out_of_range if idx > content.size()
            pointer_t outptr = content.at(idx);
            content.erase(
                std::remove_if(content.begin(), content.end(),
                      [outptr](pointer_t p) { return p == outptr || p == nullptr; }),
                content.end());
            content.shrink_to_fit();
            return unique_t(outptr);
        }
        
        unique_t pop() {
            pointer_t outptr = content.back();
            content.pop_back();
            return unique_t(outptr);
        }
        
        void reset() {
            vector_size_t idx = 0,
                          max = content.size();
            for (; idx != max; ++idx) { delete content[idx]; }
        }
        void reset(vector_t&& vector) {
            reset();
            content = std::move(vector);
        }
        
        ~ImageList() { reset(); }
        
        /// After calling release(), ownership of the content image ponters
        /// is transferred to the caller, who must figure out how to delete them.
        /// Also note that release() resets the internal vector.
        vector_t release() {
            vector_t out;
            out.swap(content);
            return out;
        }
        
        void swap(ImageList& other) noexcept {
            content.swap(other.content);
        }
        
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
    
    // template <>
    // struct hash<filesystem::path> {
    //
    //     typedef filesystem::path argument_type;
    //     typedef std::size_t result_type;
    //
    //     result_type operator()(argument_type const& p) const {
    //         return static_cast<result_type>(p.hash());
    //     }
    //
    // };
    
}; /* namespace std */


#endif /// LIBIMREAD_IMAGE_HH_