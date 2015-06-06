/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEFORMAT_HH_
#define LIBIMREAD_IMAGEFORMAT_HH_

#include <cstdint>
#include <vector>
#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/image.hh>
#include <libimread/options.hh>

namespace im {
    
    inline bool match_magic(byte_source *src, const char *magic, const std::size_t n) {
        if (!src->can_seek()) { return false; }
        std::vector<byte> buf;
        buf.resize(n);
        const int n_read = src->read(&buf.front(), n);
        src->seek_relative(-n_read);
        return (n_read == n && std::memcmp(&buf.front(), magic, n) == 0);
    }
    inline bool match_magic(byte_source *src, const std::string &magic, const std::size_t n) {
        return match_magic(src, magic.c_str(), n);
    }
    
    
    class ImageFormat {
        public:
            typedef std::false_type can_read;
            typedef std::false_type can_read_multi;
            typedef std::false_type can_read_metadata;
            typedef std::false_type can_write;
            typedef std::false_type can_write_multi;
            typedef std::false_type can_write_metadata;
            
            virtual ~ImageFormat() {}
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual std::unique_ptr<image_list> read_multi(byte_source *src,
                                                           ImageFactory *factory,
                                                           const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write_multi(std::vector<Image> &input,
                                     byte_sink* output,
                                     const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
    };

}

#endif /// LIBIMREAD_IMAGEFORMAT_HH_