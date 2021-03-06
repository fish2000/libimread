/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_EXECUTE_H_
#define LIBIMREAD_EXT_FILESYSTEM_EXECUTE_H_

#include <string>

namespace filesystem {
    
    namespace detail {
        
        static constexpr int EXECUTION_BUFFER_SIZE = 2048;
        
        std::string execute(char const* command,
                            char const* workingdir = nullptr);
        
    }
    
}

#endif /// LIBIMREAD_EXT_FILESYSTEM_EXECUTE_H_