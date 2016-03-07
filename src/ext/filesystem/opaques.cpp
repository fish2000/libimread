
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    namespace detail {
        
        filesystem::directory ddopen(char const* c) {
            return filesystem::directory(::opendir(path::absolute(c).c_str()));
        }
        
        filesystem::directory ddopen(std::string const& s) {
            return filesystem::directory(::opendir(path::absolute(s).c_str()));
        }
        
        filesystem::directory ddopen(path const& p) {
            return filesystem::directory(::opendir(p.make_absolute().c_str()));
        }
        
        inline char const* fm(mode m) noexcept { return m == mode::READ ? "r+b" : "w+x"; }
        
        filesystem::file ffopen(std::string const& s, mode m) {
            return filesystem::file(std::fopen(s.c_str(), fm(m)));
        }
        
    } /* namespace detail */

} /* namespace filesystem */