/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_IMAGEFORMAT_HH_
#define LIBIMREAD_INCLUDE_IMAGEFORMAT_HH_

#include <cstdint>
#include <string>
#include <memory>
#include <unordered_map>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/iod.hh>
#include <libimread/ext/base64.hh>
#include <libimread/symbols.hh>
#include <libimread/traits.hh>

namespace im {
    
    /// forward declarations
    class Image;
    class ImageFactory;
    struct ImageList;
    
    class Metadata;
    class byte_source;
    class byte_sink;
    struct Options;
    struct OptionsList;
    
    bool match_magic(byte_source*, char const*, std::size_t const);
    bool match_magic(byte_source*, std::string const&);
    
    #define DECLARE_BASE_OPTIONS(...)                                                  \
        using options_t = decltype(D(__VA_ARGS__));                                    \
        static ImageFormat::unique_t create();                                         \
        static const options_t OPTS() {                                                \
            const options_t O = D(__VA_ARGS__);                                        \
            return O;                                                                  \
        }                                                                              \
        static const std::string classname;                                            \
        static const options_t options;                                                \
        static const capacity_t capacity;
    
    /// use `DECLARE_OPTIONS("value", "another-value", ...);` in format.hh:
    
    #define DECLARE_OPTIONS(...)                                                       \
        DECLARE_BASE_OPTIONS(__VA_ARGS__);                                             \
        virtual Options get_options() const override;                                  \
        virtual std::size_t hash() const override;
    
    #define SIGNATURE(bytes, length)                                                   \
        D(_bytes    = base64::encode(bytes, length),                                   \
          _length   = length)
    
    /// ... then use `DECLARE_FORMAT_OPTIONS(FormatClassName);` in format.cpp:
    
    #define DECLARE_FORMAT_OPTIONS(format)                                             \
        ImageFormat::unique_t format::create() {                                       \
            return std::make_unique<format>();                                         \
        }                                                                              \
        const std::string format::classname = #format;                                 \
        const format::options_t format::options = format::OPTS();                      \
        const format::capacity_t format::capacity = format::CAPACITY();                \
        Options format::get_options() const {                                          \
            return Options::parse(iod::json_encode(iod::cat(format::options,           \
                                              D(_capacity = format::capacity))));      \
        }                                                                              \
        std::size_t format::hash() const {                                             \
            std::hash<std::string> hasher;                                             \
            return hasher(format::classname);                                          \
        }                                                                              \
        namespace {                                                                    \
            ImageFormat::Registrar<format> format##Registrar(                          \
                                           format::options.suffixes[0]);               \
        }
    
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
            using create_f      = unique_t(void);
            using create_t      = std::add_pointer_t<create_f>;
            using registry_t    = std::unordered_map<std::string, create_t>;
            
            using capacity_t    = decltype(D(
                _can_read(_optional,            _json_key = _can_read)           = bool(),
                _can_read_multi(_optional,      _json_key = _can_read_multi)     = bool(),
                _can_read_metadata(_optional,   _json_key = _can_read_metadata)  = bool(),
                _can_write(_optional,           _json_key = _can_write)          = bool(),
                _can_write_multi(_optional,     _json_key = _can_write_multi)    = bool(),
                _can_write_metadata(_optional,  _json_key = _can_write_metadata) = bool()
            ));
                
            static const capacity_t CAPACITY() {
                const capacity_t C(
                    false, false, false,
                    false, false, false
                );
                return C;
            }
            
            DECLARE_BASE_OPTIONS(
                _signatures = { SIGNATURE("xxxxxxxx", 8) },
                _suffixes = { "imr", "imread" },
                _mimetype = "application/octet-stream"
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
            
            virtual std::string get_mimetype() const;
            virtual std::string get_suffix() const;
            virtual std::string get_suffix(bool with_period) const;
            
            /// SPOILER ALERT:
            /// static-const options_t member 'options' declared by DECLARE_OPTIONS()
            virtual Options get_options() const;
            virtual Options add_options(Options const& opts) const;
            virtual std::size_t hash() const;
            
            virtual ~ImageFormat();
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts);
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         Options const& opts);
            
            virtual Metadata read_metadata(byte_source* src,
                                           Options const& opts);
            
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts);
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     Options const& opts);
            
            virtual bool format_can_read() const noexcept;
            virtual bool format_can_read_multi() const noexcept;
            virtual bool format_can_read_metadata() const noexcept;
            virtual bool format_can_write() const noexcept;
            virtual bool format_can_write_multi() const noexcept;
            virtual bool format_can_write_metadata() const noexcept;
            
            static registry_t& registry();
            
            bool operator==(ImageFormat const&) const;
            bool operator!=(ImageFormat const&) const;
    
    };
    
    template <typename FormatType>
    class ImageFormatBase : public ImageFormat {
        public:
            static const capacity_t CAPACITY() {
                const capacity_t C(
                    (traits::has_read<FormatType>()          ),
                    (traits::has_read_multi<FormatType>()    ),
                    (traits::has_read_metadata<FormatType>() ),
                    (traits::has_write<FormatType>()         ),
                    (traits::has_write_multi<FormatType>()   ),
                    (traits::has_write_metadata<FormatType>())
                );
                return C;
            }
            
            static bool match_format(byte_source* src) {
                return match_magic(src,
                    base64::decode(FormatType::options.signatures[0].bytes).get(),
                                   FormatType::options.signatures[0].length);
            }
            
            static std::string suffix() {
                return FormatType::options.suffixes[0];
            }
            
            static std::string suffix(bool with_period) {
                return with_period ? (std::string(".") + FormatType::options.suffixes[0]) :
                                                         FormatType::options.suffixes[0];
            }
            
            static std::string mimetype() {
                return FormatType::options.mimetype;
            }
            
            virtual std::string get_suffix() const override {
                return FormatType::options.suffixes[0];
            }
            
            virtual std::string get_suffix(bool with_period) const override {
                return with_period ? (std::string(".") + FormatType::options.suffixes[0]) :
                                                         FormatType::options.suffixes[0];
            }
            
            virtual std::string get_mimetype() const override {
                return FormatType::options.mimetype;
            }
            
            virtual bool format_can_read() const noexcept override           { return FormatType::can_read::value;           }
            virtual bool format_can_read_multi() const noexcept override     { return FormatType::can_read_multi::value;     }
            virtual bool format_can_read_metadata() const noexcept override  { return FormatType::can_read_metadata::value;  }
            virtual bool format_can_write() const noexcept override          { return FormatType::can_write::value;          }
            virtual bool format_can_write_multi() const noexcept override    { return FormatType::can_write_multi::value;    }
            virtual bool format_can_write_metadata() const noexcept override { return FormatType::can_write_metadata::value; }
    
    };
    
    /// based on im::detail::are_bases_of<…> in traits.hh:
    template <typename ...Types>
    struct is_imageformat : detail::are_bases_of<im::ImageFormat, Types...>::type
    {};
    
    template <typename ...Types>
    constexpr bool is_imageformat_v = is_imageformat<Types...>::value;
    
} /* namespace im */

template <typename T, typename U>
std::enable_if_t<
  im::is_imageformat_v<T, U>,
    bool> operator==(T&& lhs, U&& rhs) {
      return std::forward<T>(lhs).hash() == std::forward<U>(rhs).hash(); }

template <typename T, typename U>
std::enable_if_t<
  im::is_imageformat_v<T, U>,
    bool> operator!=(T&& lhs, U&& rhs) {
      return std::forward<T>(lhs).hash() != std::forward<U>(rhs).hash(); }

namespace std {
    
    /// std::hash specialization for im::ImageFormat
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::ImageFormat> {
        
        typedef im::ImageFormat argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const&) const;
        
    };
    
} /* namespace std */

#endif /// LIBIMREAD_INCLUDE_IMAGEFORMAT_HH_