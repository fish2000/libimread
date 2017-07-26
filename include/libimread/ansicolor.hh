/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ANSICOLORS_HH_
#define LIBIMREAD_ANSICOLORS_HH_

#include <string>
#include <iostream>

namespace ansi {
    
    enum class code : std::size_t {
        FM_RESET            = 0,
        FM_BOLD             = 1,
        FM_DIM              = 2,
        FM_UNDERLINE        = 4,
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
    
    class ANSI final {
        
        public:
            explicit ANSI(code);
            
        public:
            friend std::ostream& operator<<(std::ostream&, ANSI const&);
        
        public:
            std::string str() const;
            std::string str(std::string const&) const;
            const char* c_str() const;
            operator std::string() const;
            operator const char*() const;
        
        public:
            std::size_t code_value() const;
        
        private:
            code m_code;
        
    };
    
    const ANSI reset            = ANSI(code::FM_RESET);
    const ANSI termdefault      = ANSI(code::FG_DEFAULT_COLOR);
    
    const ANSI bold             = ANSI(code::FM_BOLD);
    const ANSI dim              = ANSI(code::FM_DIM);
    const ANSI underline        = ANSI(code::FM_UNDERLINE);
    
    const ANSI red              = ANSI(code::FG_RED);
    const ANSI green            = ANSI(code::FG_GREEN);
    const ANSI yellow           = ANSI(code::FG_YELLOW);
    const ANSI blue             = ANSI(code::FG_BLUE);
    const ANSI magenta          = ANSI(code::FG_MAGENTA);
    const ANSI cyan             = ANSI(code::FG_CYAN);
    
    const ANSI lightred         = ANSI(code::FG_LIGHTRED);
    const ANSI lightgreen       = ANSI(code::FG_LIGHTGREEN);
    const ANSI lightyellow      = ANSI(code::FG_LIGHTYELLOW);
    const ANSI lightblue        = ANSI(code::FG_LIGHTBLUE);
    const ANSI lightmagenta     = ANSI(code::FG_LIGHTMAGENTA);
    const ANSI lightcyan        = ANSI(code::FG_LIGHTCYAN);
    
    const ANSI lightgray        = ANSI(code::FG_LIGHTGRAY);
    const ANSI darkgray         = ANSI(code::FG_DARKGRAY);
    const ANSI white            = ANSI(code::FG_WHITE);
    
}

#endif /// LIBIMREAD_ANSICOLORS_HH_