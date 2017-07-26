
#include <array>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>

namespace filesystem {
    
    namespace detail {
        
        directory ddopen(char const* c) {
            return directory(::opendir(path::absolute(c).c_str()));
        }
        
        directory ddopen(std::string const& s) {
            return directory(::opendir(path::absolute(s).c_str()));
        }
        
        directory ddopen(path const& p) {
            return directory(::opendir(p.make_absolute().c_str()));
        }
        
        inline char const* fm(mode m) noexcept {
            return m == mode::READ ? "r+b" : "w+x";
        }
        
        inline int dm(mode m) noexcept {
            return m == mode::READ ? O_RDWR | O_NONBLOCK | O_CLOEXEC :
                                     O_RDWR | O_NONBLOCK | O_CLOEXEC | O_CREAT | O_EXCL | O_TRUNC;
        }
        
        filesystem::file ffopen(std::string const& s, mode m) {
            return filesystem::file(std::fopen(s.c_str(), fm(m)));
        }
        
    } /* namespace detail */

} /* namespace filesystem */