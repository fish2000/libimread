/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <vector>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/imageformat.hh>
#include <iod/json.hh>

namespace im {
    
    bool match_magic(byte_source* src, const char* magic, const std::size_t n) {
        if (!src->can_seek()) { return false; }
        std::vector<byte> buf;
        buf.resize(n);
        const int n_read = static_cast<int>(src->read(&buf.front(), n));
        src->seek_relative(-n_read);
        return (n_read == n && std::memcmp(&buf.front(), magic, n) == 0);
    }
    
    bool match_magic(byte_source* src, const std::string& magic) {
        return match_magic(src, magic.c_str(), magic.size());
    }
    
    DECLARE_FORMAT_OPTIONS(ImageFormat);
    
    /// including <iod/json.hh> along with Halide.h will cause a conflict --
    /// -- some macro called `user_error` I believe -- that won't compile.
    /// This static method is defined in here for this reason.
    options_map ImageFormat::encode_options(options_t which_options) {
        return options_map::parse(iod::json_encode(which_options));
    }

    
}
