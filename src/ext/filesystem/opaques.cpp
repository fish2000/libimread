
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    namespace {
        
        __attribute__((always_inline))
        static char const* fm(mode m) noexcept {
            return m == mode::READ ? "r+b" : "w+x";
        }
        
        __attribute__((always_inline))
        static int dm(mode m) noexcept {
            return m == mode::READ ? O_RDWR | O_NONBLOCK | O_CLOEXEC :
                                     O_RDWR | O_NONBLOCK | O_CLOEXEC | O_CREAT | O_EXCL | O_TRUNC;
        }
        
    } /// namespace (anon.)
    
    namespace detail {
        
        using filesystem::path;
        
        filesystem::directory ddopen(path const& p) {
            return filesystem::directory(::opendir(p.make_absolute().c_str()));
        }
        
        filesystem::directory ddopen(std::string const& s) {
            return filesystem::directory(::opendir(path::absolute(s).c_str()));
        }
        
        filesystem::directory ddopen(int const descriptor) {
            return filesystem::directory(::fdopendir(descriptor));
        }
        
        filesystem::file ffopen(path const& p, mode m) {
            return filesystem::file(std::fopen(p.normalize().c_str(), fm(m)));
        }
        
        filesystem::file ffopen(std::string const& s, mode m) {
            return filesystem::file(std::fopen(path::normalize(s).c_str(), fm(m)));
        }
        
        filesystem::file ffopen(int const descriptor, mode m) {
            return filesystem::file(::fdopen(descriptor, fm(m)));
        }
        
        namespace shared {
            
            filesystem::shared::directory ddopen(path const& p) {
                return filesystem::shared::directory(::opendir(p.make_absolute().c_str()),
                                               detail::closer<DIR>());
            }
            
            filesystem::shared::directory ddopen(std::string const& s) {
                return filesystem::shared::directory(::opendir(path::absolute(s).c_str()),
                                               detail::closer<DIR>());
            }
            
            filesystem::shared::directory ddopen(int const descriptor) {
                return filesystem::shared::directory(::fdopendir(descriptor),
                                               detail::closer<DIR>());
            }
            
            filesystem::shared::file ffopen(path const& p, mode m) {
                return filesystem::shared::file(std::fopen(p.normalize().c_str(), fm(m)),
                                             detail::closer<FILE>());
            }
            
            filesystem::shared::file ffopen(std::string const& s, mode m) {
                return filesystem::shared::file(std::fopen(path::normalize(s).c_str(), fm(m)),
                                             detail::closer<FILE>());
            }
            
            filesystem::shared::file ffopen(int const descriptor, mode m) {
                return filesystem::shared::file(::fdopen(descriptor, fm(m)),
                                          detail::closer<FILE>());
            }
            
        } /// namespace shared
        
    } /// namespace detail
    
} /// namespace filesystem