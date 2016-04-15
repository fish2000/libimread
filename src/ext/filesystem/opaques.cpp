
#include <fcntl.h>
#include <unistd.h>
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
        inline int dm(mode m) noexcept { return m == mode::READ ? O_RDONLY | O_NONBLOCK :
                                                                  O_WRONLY | O_NONBLOCK | O_CREAT | O_EXCL | O_TRUNC; }
        
        filesystem::file ffopen(std::string const& s, mode m) {
            return filesystem::file(std::fopen(s.c_str(), fm(m)));
        }
        
        // filesystem::descriptor fdopen(std::string const& s, mode m) {
        //     return filesystem::descriptor(::open(s.c_str(), dm(m)));
        // }
        
    } /* namespace detail */

} /* namespace filesystem */