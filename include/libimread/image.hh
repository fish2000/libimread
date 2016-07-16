/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGE_HH_
#define LIBIMREAD_IMAGE_HH_

#include <vector>
#include <memory>
#include <string>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/pixels.hh>

namespace im {
    
    class Histogram;
    
    class Image {
        
        public:
            using unique_image_t = std::unique_ptr<Image>;
            using shared_image_t = std::shared_ptr<Image>;
            using weak_image_t   = std::weak_ptr<Image>;
            using image_ptr_t    = std::add_pointer_t<Image>;
            
            using const_unique_image_t = std::unique_ptr<Image const>;
            using const_shared_image_t = std::shared_ptr<Image const>;
            using const_weak_image_t   = std::weak_ptr<Image const>;
            using const_image_ptr_t    = std::add_pointer_t<Image const>;
            
            virtual ~Image() {}
            
            virtual void* rowp(int r) const = 0;
            virtual void* rowp() const;
            virtual int nbits() const = 0;
            
            virtual int nbytes() const;
            virtual int ndims() const = 0;
            virtual int dim(int) const = 0;
            virtual int stride(int) const = 0;
            virtual int min(int) const;
            virtual bool is_signed() const = 0;
            virtual bool is_floating_point() const = 0;
            
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
            virtual int otsu() const;
            
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
                planevec_t out;
                out.resize(siz, 0);
                for (int x = 0; x < w; ++x) {
                    for (int y = 0; y < h; ++y) {
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
                pixvec_t out;
                for (int idx = 0; idx < planecount; ++idx) {
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
    };
    
    class ImageWithMetadata {
        
        public:
            using bytevec_t = std::vector<byte>;
            
            ImageWithMetadata();
            ImageWithMetadata(std::string const& m);
            virtual ~ImageWithMetadata();
            
            bool has_meta() const;
            std::string const& get_meta() const;
            std::string const& set_meta(std::string const&);
            
            bool has_icc_name() const;
            std::string const& get_icc_name() const;
            std::string const& set_icc_name(std::string const&);
            
            bool has_icc_data() const;
            bytevec_t const& get_icc_data() const;
            bytevec_t const& set_icc_data(bytevec_t const&);
            bytevec_t const& set_icc_data(byte*, std::size_t);
            
        protected:
            std::string meta;
            std::string icc_name;
            bytevec_t icc_data;
    };
    
} /* namespace im */

#endif /// LIBIMREAD_IMAGE_HH_