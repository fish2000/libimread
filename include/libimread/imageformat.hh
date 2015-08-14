/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEFORMAT_HH_
#define LIBIMREAD_IMAGEFORMAT_HH_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/image.hh>
#include <libimread/symbols.hh>
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
    inline bool match_magic(byte_source *src, const std::string &magic) {
        return match_magic(src, magic.c_str(), magic.size());
    }
    
    class ImageFormat {
        
        public:
            using can_read              = std::false_type;
            using can_read_multi        = std::false_type;
            using can_read_metadata     = std::false_type;
            using can_write             = std::false_type;
            using can_write_multi       = std::false_type;
            using can_write_metadata    = std::false_type;
            
            using options_t             = decltype(D(
                _signature(_optional, _json_key = _signature)  = std::string(),
                _suffix(_optional, _json_key = _suffix)        = std::string()
            ));
            
            static const options_t OPTS() {
                const options_t O(
                    "xxxxxxxx",         /// signature
                    "image"             /// suffix
                );
                return O;
            }
            
            static const options_t options;
            
            /// NOT AN OVERRIDE:
            static bool match_format(byte_source *src) {
                return match_magic(src, options.signature);
            }
            
            static std::string get_suffix() {
                return std::string(options.suffix);
            }
            
            virtual ~ImageFormat() {}
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual std::unique_ptr<ImageList> read_multi(byte_source *src,
                                                           ImageFactory *factory,
                                                           const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write_multi(ImageList &input,
                                     byte_sink* output,
                                     const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
    };

}

#endif /// LIBIMREAD_IMAGEFORMAT_HH_