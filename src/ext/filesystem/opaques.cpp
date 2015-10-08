
#include <libimread/ext/filesystem/opaques.h>

namespace filesystem {
    
    namespace detail {
        
        filesystem::directory ddopen(const char *c) {
            return filesystem::directory(::opendir(path::absolute(c).c_str()));
        }
        
        filesystem::directory ddopen(const std::string &s) {
            return filesystem::directory(::opendir(path::absolute(s).c_str()));
        }
        
        filesystem::directory ddopen(const path &p) {
            return filesystem::directory(::opendir(p.make_absolute().c_str()));
        }
        
        inline const char *fm(mode m) noexcept { return m == mode::READ ? "r+b" : "w+x"; }
        
        filesystem::file ffopen(const std::string &s, mode m) {
            return filesystem::file(std::fopen(s.c_str(), fm(m)));
        }
        
    } /* namespace detail */

} /* namespace filesystem */