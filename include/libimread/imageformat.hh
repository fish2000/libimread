/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IMAGEFORMAT_HH_
#define LIBIMREAD_IMAGEFORMAT_HH_

#include <cstdint>
#include <string>
#include <memory>
#include <unordered_map>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>
#include <libimread/image.hh>
#include <libimread/symbols.hh>
#include <libimread/options.hh>

namespace im {
    
    bool match_magic(byte_source*, char const*, std::size_t const);
    bool match_magic(byte_source*, std::string const&);
    
    /// use `DECLARE_OPTIONS("value", "another-value", ...);` in format.hh:
    
    #define DECLARE_OPTIONS(...)                                                            \
        static ImageFormat::unique_t create();                                              \
        static const options_t OPTS() {                                                     \
            const options_t O(__VA_ARGS__);                                                 \
            return O;                                                                       \
        }                                                                                   \
        static const options_t options;
    
    /// ... then use `DECLARE_FORMAT_OPTIONS(FormatClassName);` in format.cpp:
    
    #define DECLARE_FORMAT_OPTIONS(format)                                                  \
        ImageFormat::unique_t format::create() {                                            \
            return std::make_unique<format>();                                              \
        }                                                                                   \
        const ImageFormat::options_t format::options = format::OPTS();                      \
        namespace {                                                                         \
            ImageFormat::Registrar<format> registrar(format::options.suffix);               \
        };
    
    /// ... those macros also set your format up to register its class (see below).
    
    class ImageFormat {
        
        public:
            using can_read              = std::false_type;
            using can_read_multi        = std::false_type;
            using can_read_metadata     = std::false_type;
            using can_write             = std::false_type;
            using can_write_multi       = std::false_type;
            using can_write_metadata    = std::false_type;
            
            using format_t      = ImageFormat;
            using unique_t      = std::unique_ptr<format_t>;
            using create_f      = unique_t();
            using create_t      = std::add_pointer_t<create_f>;
            using registry_t    = std::unordered_map<std::string, create_t>;
            
            using options_t     = decltype(D(
                _signature(_optional, _json_key = _signature)  = std::string(),
                _suffix(_optional,    _json_key = _suffix)     = std::string(),
                _mimetype(_optional,  _json_key = _mimetype)   = std::string()
            ));
            
            DECLARE_OPTIONS(
                "xxxxxxxx",                                     /// signature
                "imread",                                       /// suffix
                "application/octet-stream"                      /// mimetype
            );
            
            /// These static methods, and the ImageFormat::Registrar<Derived> template,
            /// implement the API to the format registry. Just use the static method:
            /// 
            ///     auto format_ptr = ImageFormat::named("jpg");
            /// 
            /// ... and you get a std::unique_ptr<ImageFormat> wrapping a pointer
            /// to a new heap-allocated instance of the named ImageFormat subclass.
            
            static void registrate(std::string const& name, create_t fp);
            static unique_t named(std::string const& name);
            
            /// Format registry derived from: http://stackoverflow.com/a/11176265/298171
            
            template <typename D>
            struct Registrar {
                explicit Registrar(std::string const& name) {
                    ImageFormat::registrate(name, &D::create);
                }
            };
            
            virtual std::string get_suffix() const;
            virtual std::string get_mimetype() const;
            
            /// SPOILER ALERT:
            /// static-const options_t member 'options' declared by DECLARE_OPTIONS()
            
            static  options_map encode_options(options_t const& opts = options);
            virtual options_map get_options() const;
            virtual options_map add_options(options_map const& opts) const;
            
            virtual ~ImageFormat();
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts);
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         options_map const& opts);
            
            virtual void write(Image& input,
                               byte_sink* output,
                               options_map const& opts);
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     options_map const& opts);
            
            virtual bool format_can_read() const noexcept;
            virtual bool format_can_read_multi() const noexcept;
            virtual bool format_can_read_metadata() const noexcept;
            virtual bool format_can_write() const noexcept;
            virtual bool format_can_write_multi() const noexcept;
            virtual bool format_can_write_metadata() const noexcept;
            
            static registry_t& registry();
    
    };
    
    template <typename FormatType>
    class ImageFormatBase : public ImageFormat {
        
        public:
            static bool match_format(byte_source* src) {
                return match_magic(src, FormatType::options.signature);
            }
            
            static std::string suffix() {
                return FormatType::options.suffix;
            }
            
            static std::string mimetype() {
                return FormatType::options.mimetype;
            }
            
            virtual std::string get_suffix() const override {
                return FormatType::options.suffix;
            }
            
            virtual std::string get_mimetype() const override {
                return FormatType::options.mimetype;
            }
            
            virtual options_map get_options() const override {
                return ImageFormat::encode_options(FormatType::options);
            }
            
            virtual options_map add_options(options_map const& opts) const override {
                options_map result = ImageFormat::encode_options(FormatType::options);
                return result.update(opts);
            }
            
            virtual bool format_can_read() const noexcept override           { return FormatType::can_read::value; }
            virtual bool format_can_read_multi() const noexcept override     { return FormatType::can_read_multi::value; }
            virtual bool format_can_read_metadata() const noexcept override  { return FormatType::can_read_metadata::value; }
            virtual bool format_can_write() const noexcept override          { return FormatType::can_write::value; }
            virtual bool format_can_write_multi() const noexcept override    { return FormatType::can_write_multi::value; }
            virtual bool format_can_write_metadata() const noexcept override { return FormatType::can_write_metadata::value; }
            
            /// LONGCAT IS LOOOOOOOOOONG erm I mean
            /// SAME CLASS IS SAME
            template <typename OtherFormatType,
                      typename X = std::enable_if_t<
                                   std::is_base_of<ImageFormat, OtherFormatType>::value>> inline
            bool operator==(OtherFormatType const& other) const noexcept {
                return std::is_same<std::remove_cv_t<FormatType>,
                                    std::remove_cv_t<OtherFormatType>>::value;
            }
            
            template <typename OtherFormatType,
                      typename X = std::enable_if_t<
                                   std::is_base_of<ImageFormat, OtherFormatType>::value>> inline
            bool operator!=(OtherFormatType const& other) const noexcept {
                return !std::is_same<std::remove_cv_t<FormatType>,
                                     std::remove_cv_t<OtherFormatType>>::value;
            }
    
    };

}

#endif /// LIBIMREAD_IMAGEFORMAT_HH_