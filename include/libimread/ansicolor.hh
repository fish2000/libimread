/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ANSICOLORS_HH_
#define LIBIMREAD_ANSICOLORS_HH_

#include <string>
#include <iostream>
#include <tuple>
#include <utility>

#include <libimread/libimread.hpp>

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
    
    extern const ANSI reset;
    extern const ANSI termdefault;
    
    extern const ANSI bold;
    extern const ANSI dim;
    extern const ANSI underline;
    
    extern const ANSI red;
    extern const ANSI green;
    extern const ANSI yellow;
    extern const ANSI blue;
    extern const ANSI magenta;
    extern const ANSI cyan;
    
    extern const ANSI lightred;
    extern const ANSI lightgreen;
    extern const ANSI lightyellow;
    extern const ANSI lightblue;
    extern const ANSI lightmagenta;
    extern const ANSI lightcyan;
    
    extern const ANSI lightgray;
    extern const ANSI darkgray;
    extern const ANSI white;
    
}

#endif /// LIBIMREAD_ANSICOLORS_HH_