/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ANSICOLORS_HH_
#define LIBIMREAD_ANSICOLORS_HH_

#include <string>
#include <iostream>

#include <libimread/libimread.hpp>
#include <libimread/stringnormatives.hh>

namespace ansi {
    
    using codeformat_t = std::pair<std::string,
                                   std::string>;
    
    using codedtext_t = std::tuple<std::string,
                                   std::string,
                                   std::string>;
    
    enum class code_t : std::size_t {
        
        /// control and meta-typographic
        /// style<†> and ink-mode<§> settings:
        FM_RESET            = 0,
        FM_BOLD             = 1,
        FM_DIM              = 2,
        FM_UNDERLINE        = 4,
        
        /// foreground (type) colors:
        FG_BLACK            = 30,
        FG_RED              = 31,
        FG_GREEN            = 32,
        FG_YELLOW           = 33,
        FG_BLUE             = 34,
        FG_MAGENTA          = 35,
        FG_CYAN             = 36,
        FG_LIGHTGRAY        = 37,
        FG_DEFAULT_COLOR    = 39,
        
        /// background (lining-space) colors:
        BG_RED              = 41,
        BG_GREEN            = 42,
        BG_BLUE             = 44,
        BG_DEFAULT          = 49,
        
        /// forground (type) variant colors:
        FG_DARKGRAY         = 90,
        FG_LIGHTRED         = 91,
        FG_LIGHTGREEN       = 92,
        FG_LIGHTYELLOW      = 93,
        FG_LIGHTBLUE        = 94,
        FG_LIGHTMAGENTA     = 95,
        FG_LIGHTCYAN        = 96,
        FG_WHITE            = 97
        
        /// [†] » underline, synthetic “bold” glyph-scaling, &c…
        /// [§] » “dimming” via HSB-model B-channel-normalize
    };
    
    class ANSI final {
        
        public:
            explicit ANSI(code_t);
            
        public:
            friend std::ostream& operator<<(std::ostream&, ANSI const&);
            friend std::ostream& operator<<(std::ostream&, codedtext_t const&);
        
        public:
            std::string str() const;
            std::string str(std::string const&) const;
            const char* c_str() const;
            operator std::string() const;
            operator const char*() const;
        
        public:
            std::size_t value() const;
            codeformat_t format() const;
            codedtext_t formatted(std::string const&) const;
            
        public:
            std::string operator()(std::string const&) const;
            
        private:
            code_t code;
        
    };
    
    const ANSI reset            = ANSI(code_t::FM_RESET);
    const ANSI termdefault      = ANSI(code_t::FG_DEFAULT_COLOR);
    
    const ANSI bold             = ANSI(code_t::FM_BOLD);
    const ANSI dim              = ANSI(code_t::FM_DIM);
    const ANSI underline        = ANSI(code_t::FM_UNDERLINE);
    
    const ANSI red              = ANSI(code_t::FG_RED);
    const ANSI green            = ANSI(code_t::FG_GREEN);
    const ANSI yellow           = ANSI(code_t::FG_YELLOW);
    const ANSI blue             = ANSI(code_t::FG_BLUE);
    const ANSI magenta          = ANSI(code_t::FG_MAGENTA);
    const ANSI cyan             = ANSI(code_t::FG_CYAN);
    
    const ANSI lightred         = ANSI(code_t::FG_LIGHTRED);
    const ANSI lightgreen       = ANSI(code_t::FG_LIGHTGREEN);
    const ANSI lightyellow      = ANSI(code_t::FG_LIGHTYELLOW);
    const ANSI lightblue        = ANSI(code_t::FG_LIGHTBLUE);
    const ANSI lightmagenta     = ANSI(code_t::FG_LIGHTMAGENTA);
    const ANSI lightcyan        = ANSI(code_t::FG_LIGHTCYAN);
    
    const ANSI lightgray        = ANSI(code_t::FG_LIGHTGRAY);
    const ANSI darkgray         = ANSI(code_t::FG_DARKGRAY);
    const ANSI white            = ANSI(code_t::FG_WHITE);
    
}

#endif /// LIBIMREAD_ANSICOLORS_HH_