
#include <libimread/libimread.hpp>
#include <libimread/file.hh>

namespace im {
    
    bool file_exists(char *path) {
        return (::access(path, R_OK) != -1);
    }
    bool file_exists(const char *path) {
        return file_exists(const_cast<char *>(path));
    }
    bool file_exists(std::string path) {
        return file_exists(path.c_str());
    }
    bool file_exists(const std::string &path) {
        return file_exists(path.c_str());
    }
    
    bool file_source_sink::exists() const { return im::file_exists(path()); }
    
}
