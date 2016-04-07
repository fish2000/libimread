/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include "preview.hpp"
#include <chrono>
#include <thread>

namespace py {
    
    namespace image {
        
        void preview(path const& timage) {
            /// INSERT GENERIC PREVIEWER AND/OR
            /// "PLATFORM UNKNOWN" MESSAGE
            /// HERE
            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    py::image::sleeptime));
            
        }
        
    }
    
}

