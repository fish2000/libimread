/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_
#define LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_

#include <libimread/libimread.hpp>

namespace filesystem {
    
    namespace detail {
        
        #ifdef IM_HAVE_AUTOFS_NOWAIT
        
        struct nowait_t {
            
            nowait_t();
            ~nowait_t();
            
            private:
                nowait_t(nowait_t const&);
                nowait_t(nowait_t&&);
                nowait_t& operator=(nowait_t const&);
                nowait_t& operator=(nowait_t&&);
                mutable int descriptor = -1;
            
        };
        
        #else
        
        struct nowait_t {};
        
        #endif
        
        
        #ifdef IM_HAVE_AUTOFS_NOTRIGGER
        
        struct notrigger_t {
            
            notrigger_t();
            ~notrigger_t();
            
            private:
                notrigger_t(notrigger_t const&);
                notrigger_t(notrigger_t&&);
                notrigger_t& operator=(notrigger_t const&);
                notrigger_t& operator=(notrigger_t&&);
                mutable int descriptor = -1;
                
        };
        
        #else
        
        struct notrigger_t {};
        
        #endif
        
    }
    
}

#endif /// LIBIMREAD_EXT_FILESYSTEM_NOWAIT_H_