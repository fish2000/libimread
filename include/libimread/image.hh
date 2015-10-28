/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/pixels.hh>

namespace im {
    class Image {
        public:
            virtual ~Image() {}
            
            virtual void* rowp(int r) = 0;
            virtual const int nbits() const = 0;
            
            virtual const int nbytes() const {
                const int bits = this->nbits();
                return (bits / 8) + bool(bits % 8);
            }
            
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            
            virtual int dim_or(int dim, int default_value) const {
                if (dim >= this->ndims()) { return default_value; }
                return this->dim(dim);
            }
            
            template <typename T> inline
            T* rowp_as(const int r) {
                return static_cast<T*>(this->rowp(r));
            }
            
            template <typename T = byte> inline
            pix::accessor<T> access() {
                return pix::accessor<T>(rowp_as<T>(0), stride(0),
                                                       stride(1),
                                                       stride(2));
            }
            
            template <typename T> inline
            std::vector<T*> allrows() const {
                std::vector<T*> rows;
                const int h = this->dim(0);
                for (int r = 0; r != h; ++r) {
                    rows.push_back(this->rowp_as<T>(r));
                }
                return rows;
            }
    };
    
    class ImageFactory {
        public:
            virtual ~ImageFactory() { }
            
            virtual std::unique_ptr<Image>
                create(int nbits,
                    int d0, int d1, int d2,
                    int d3=-1, int d4=-1) = 0;
            
            virtual std::shared_ptr<Image>
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
        using vector_t = std::vector<Image*>;
        using vector_size_t = vector_t::size_type;
        
        ImageList() {}
        
        vector_size_t size() const {
            return content.size();
        }
        
        void push_back(std::unique_ptr<Image> p) {
            content.push_back(p.get());
        }
        void push_back(Image* p) {
            content.push_back(p);
        }
        
        ~ImageList() {
            unsigned i = 0,
                     x = static_cast<unsigned>(content.size());
            for (; i != x; ++i) { delete content[i]; }
        }
        
        /// After calling release(), ownership of the content image ponters
        /// is transferred to the caller, who must figure out how to delete them.
        /// Also note that release() resets the internal vector.
        vector_t release() {
            vector_t out;
            out.swap(content);
            return out;
        }
        
        private:
            ImageList(ImageList&&);
            ImageList(const ImageList&);
            ImageList &operator=(ImageList&&);
            ImageList &operator=(const ImageList&);
            vector_t content;
    };
}

#endif /// LIBIMREAD_IMAGE_HH_