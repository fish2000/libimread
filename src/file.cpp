/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

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
    
    constexpr int file_source_sink::READ_FLAGS;
    constexpr int file_source_sink::WRITE_FLAGS;
    
    bool file_source_sink::exists() const { return im::file_exists(path()); }
    
}
