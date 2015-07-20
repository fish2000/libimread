/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/file.hh>

namespace im {
    
    constexpr int file_source_sink::READ_FLAGS;
    constexpr int file_source_sink::WRITE_FLAGS;
    
    bool file_source_sink::exists() const { return pth.exists(); }
    
}
