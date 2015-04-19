/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>
#include <libimread/libimread.hpp>
#include <libimread/tools.hh>

namespace im {
    class Image {
        public:
            virtual ~Image() { }
            
            virtual void *rowp(int r) = 0;
            virtual int nbits() const = 0;
            
            virtual int nbytes() const {
                const int bits = this->nbits();
                return (bits / 8) + bool(bits % 8);
            }
            
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            
            virtual int dim_or(int dim, int def) const {
                if (dim >= this->ndims()) { return def; }
                return this->dim(dim);
            }
            
            template <typename T>
            T *rowp_as(const int r) {
                return static_cast<T*>(this->rowp(r));
            }
            
            template <typename T>
            inline std::vector<T*> allrows() {
                std::vector<T*> res;
                const int h = this->dim(0);
                for (int r = 0; r != h; ++r) {
                    res.push_back(this->rowp_as<T>(r));
                }
                return res;
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
            ImageWithMetadata(const std::string &m)
                :meta(std::string(m))
                {}
            
            virtual ~ImageWithMetadata() {}
            
            std::string get_meta() { return meta; }
            void set_meta(const std::string &m) { meta = std::string(m); }
            
        private:
            std::string meta;
    };
    
    /// This class *owns* its members and will delete them if destroyed
    struct image_list {
        public:
            image_list() { }
            ~image_list() {
                for (unsigned i = 0; i != content.size(); ++i) {
                    delete content[i];
                }
            }
            
            std::vector<Image*>::size_type size() const { return content.size(); }
            void push_back(std::unique_ptr<Image> p) { content.push_back(p.release()); }
            
            /// After release(), all of the pointers will be owned by the caller
            /// who must figure out how to delete them. Note that release() resets the list.
            std::vector<Image*> release() {
                std::vector<Image*> r;
                r.swap(content);
                return r;
            }
        
        private:
            image_list(const image_list&);
            image_list &operator=(const image_list&);
            std::vector<Image*> content;
    };
}

#endif /// LIBIMREAD_IMAGE_HH_