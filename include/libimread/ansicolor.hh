/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ANSICOLORS_HH_
#define LIBIMREAD_ANSICOLORS_HH_

#include <cstring>
#include <string>
#include <iostream>
#include <libimread/libimread.hpp>

#ifndef IM_COLOR_TRACE
#define IM_COLOR_TRACE 0
#endif

namespace ansi {
    
    enum ANSICode {
        FM_RESET            = 0,
        FM_BOLD             = 1,
        FM_DIM              = 2,
        FM_UNDERLINE        = 4,
        
        // FM_BOLD_OFF      = 1,
        // FM_DIM_OFF       = 2,
        // FM_UNDERLINE_OFF = 4,
        
        FG_BLACK            = 30,
        FG_RED              = 31,
        FG_GREEN            = 32,
        FG_YELLOW           = 33,
        FG_BLUE             = 34,
        FG_MAGENTA          = 35,
        FG_CYAN             = 36,
        FG_LIGHTGRAY        = 37,
        FG_DEFAULT_COLOR    = 39,
        BG_RED              = 41,
        BG_GREEN            = 42,
        BG_BLUE             = 44,
        BG_DEFAULT          = 49,
        FG_DARKGRAY         = 90,
        FG_LIGHTRED         = 91,
        FG_LIGHTGREEN       = 92,
        FG_LIGHTYELLOW      = 93,
        FG_LIGHTBLUE        = 94,
        FG_LIGHTMAGENTA     = 95,
        FG_LIGHTCYAN        = 96,
        FG_WHITE            = 97
    };
    
    class ANSI {
        private:
            ANSICode code;
        
        public:
            explicit constexpr ANSI(ANSICode c)
                :code(c)
                {}
            
            friend std::ostream&
            operator<<(std::ostream& os, const ANSI& ansi) {
                #if IM_COLOR_TRACE == 1
                    return os << "\033[" << ansi.code << "m";
                #else
                    return os;
                #endif
            }
            
            std::string str() const {
                #if IM_COLOR_TRACE == 1
                    return "\033[" + std::to_string(code) + "m";
                #else
                    return "";
                #endif
            }
            
            const char *c_str() const {
                #if IM_COLOR_TRACE == 1
                    return str().c_str();
                #else
                    return "";
                #endif
            }
            
            inline operator std::string() const { return str(); }
            inline operator const char*() const { return c_str(); }
    };
    
    const ANSI reset            = ANSI(FM_RESET);
    const ANSI termdefault      = ANSI(FG_DEFAULT_COLOR);
    
    const ANSI bold             = ANSI(FM_BOLD);
    const ANSI dim              = ANSI(FM_DIM);
    const ANSI underline        = ANSI(FM_UNDERLINE);
    
    // const ANSI bold_off      = ANSI(FM_BOLD);
    // const ANSI dim_off       = ANSI(FM_DIM);
    // const ANSI underline_off = ANSI(FM_UNDERLINE);
    
    const ANSI red              = ANSI(FG_RED);
    const ANSI green            = ANSI(FG_GREEN);
    const ANSI yellow           = ANSI(FG_YELLOW);
    const ANSI blue             = ANSI(FG_BLUE);
    const ANSI magenta          = ANSI(FG_MAGENTA);
    const ANSI cyan             = ANSI(FG_CYAN);
    
    const ANSI lightred         = ANSI(FG_LIGHTRED);
    const ANSI lightgreen       = ANSI(FG_LIGHTGREEN);
    const ANSI lightyellow      = ANSI(FG_LIGHTYELLOW);
    const ANSI lightblue        = ANSI(FG_LIGHTBLUE);
    const ANSI lightmagenta     = ANSI(FG_LIGHTMAGENTA);
    const ANSI lightcyan        = ANSI(FG_LIGHTCYAN);
    
    const ANSI lightgray        = ANSI(FG_LIGHTGRAY);
    const ANSI darkgray         = ANSI(FG_DARKGRAY);
    const ANSI white            = ANSI(FG_WHITE);
    
}

#endif /// LIBIMREAD_ANSICOLORS_HH_