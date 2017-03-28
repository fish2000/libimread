/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <vector>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/imageformat.hh>
#include <libimread/image.hh>
#include <libimread/imagelist.hh>
#include <iod/json.hh>

namespace im {
    
    bool match_magic(byte_source* src, char const* magic, std::size_t const n) {
        std::vector<byte> buf(n);
        const int n_read = static_cast<int>(src->read(&buf.front(), n));
        src->seek_relative(-n_read);
        return (n_read == n && std::memcmp(&buf.front(), magic, n) == 0);
    }
    
    bool match_magic(byte_source* src, std::string const& magic) {
        return match_magic(src, magic.c_str(), magic.size());
    }
    
    /// initialize base class' static-const options_t member 'options' ...
    DECLARE_FORMAT_OPTIONS(ImageFormat);
    
    /// ... containing data used by format-specific functions e.g. these:
    std::string ImageFormat::get_mimetype() const { return ImageFormat::options.mimetype; }
    std::string ImageFormat::get_suffix() const   { return ImageFormat::options.suffixes[0]; }
    std::string ImageFormat::get_suffix(bool with_period) const   {
        return with_period ? (std::string(".") + ImageFormat::options.suffixes[0]) :
                                                 ImageFormat::options.suffixes[0];
    }
    
    void ImageFormat::registrate(std::string const& name, ImageFormat::create_t fp) {
        if (name.size() > 0) {
            registry()[name] = fp;
        }
    }
    
    ImageFormat::unique_t ImageFormat::named(std::string const& name) {
        auto it = registry().find(name);
        return it == registry().end() ? nullptr : (it->second)();
    }
    
    /// including <iod/json.hh> along with Halide.h will cause a conflict --
    /// -- some macro called `user_error` I believe -- that won't compile.
    /// So this next method must be defined out-of-line, in a TU set up to safely call
    /// `iod::json_encode()` (as the aforementioned conflicty include file declares it)
    options_map ImageFormat::add_options(options_map const& opts) const {
        return ImageFormat::get_options().update(opts);
    }
    
    ImageFormat::~ImageFormat() {}
    
    std::unique_ptr<Image> ImageFormat::read(byte_source* src,
                                             ImageFactory* factory,
                                             options_map const& opts) {
        imread_raise_default(NotImplementedError);
    }
    
    ImageList ImageFormat::read_multi(byte_source* src,
                                      ImageFactory* factory,
                                      options_map const& opts) {
        imread_raise_default(NotImplementedError);
    }
    
    void ImageFormat::write(Image& input,
                            byte_sink* output,
                            options_map const& opts) {
        imread_raise_default(NotImplementedError);
    }
    
    void ImageFormat::write_multi(ImageList& input,
                                  byte_sink* output,
                                  options_map const& opts) {
        imread_raise_default(NotImplementedError);
    }
    
    bool ImageFormat::format_can_read() const noexcept           { return false; }
    bool ImageFormat::format_can_read_multi() const noexcept     { return false; }
    bool ImageFormat::format_can_read_metadata() const noexcept  { return false; }
    bool ImageFormat::format_can_write() const noexcept          { return false; }
    bool ImageFormat::format_can_write_multi() const noexcept    { return false; }
    bool ImageFormat::format_can_write_metadata() const noexcept { return false; }
    
    ImageFormat::registry_t& ImageFormat::registry() {
        static ImageFormat::registry_t registry_impl;
        return registry_impl;
    }
}
