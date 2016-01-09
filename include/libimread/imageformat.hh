/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEFORMAT_HH_
#define LIBIMREAD_IMAGEFORMAT_HH_

#include <cstdint>
#include <string>
#include <memory>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/image.hh>
#include <libimread/symbols.hh>
#include <libimread/options.hh>

namespace im {
    
    bool match_magic(byte_source*, const char*, const std::size_t);
    bool match_magic(byte_source*, const std::string&);
    
    #define DECLARE_OPTIONS(...)                                                    \
        static const options_t OPTS() {                                             \
            const options_t O(__VA_ARGS__);                                         \
            return O;                                                               \
        }                                                                           \
        static const options_t options;
    
    #define DECLARE_FORMAT_OPTIONS(format)                                          \
        const ImageFormat::options_t format::options = format::OPTS();
    
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
                _suffix(_optional,    _json_key = _suffix)     = std::string(),
                _mimetype(_optional,  _json_key = _mimetype)   = std::string()
            ));
            
            DECLARE_OPTIONS(
                "xxxxxxxx",                 /// signature
                "image",                    /// suffix
                "application/octet-stream"  /// mimetype
            );
            
            /// NOT AN OVERRIDE:
            static bool match_format(byte_source*);
            static std::string get_suffix();
            static options_map get_options();
            virtual ~ImageFormat() {}
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map &opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual std::unique_ptr<ImageList> read_multi(byte_source* src,
                                                          ImageFactory* factory,
                                                          const options_map& opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) {
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     const options_map& opts) {
                imread_raise_default(NotImplementedError);
            }
            
    };

}

#endif /// LIBIMREAD_IMAGEFORMAT_HH_