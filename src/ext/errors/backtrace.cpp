/// Copyright 2014-2018 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from http://stackoverflow.com/a/31633962/298171

#include <string>
#include <libimread/ext/pystring.hh>
#include <libimread/ext/errors/demangle.hh>
#include <libimread/ext/errors/backtrace.hh>

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <ostream>

#define UNW_LOCAL_ONLY
#include <libunwind.h>

namespace terminator {
    
    namespace {
        
        static const std::string string_name = "std::string";
        static const std::string string_tpl = "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >";
        
        std::string postprocess(char const* symbolname) {
            /// call it std::string when it's a fucking std::string ok?
            return pystring::replace(symbolname, string_tpl, string_name);
        }
        
        void print_reg(std::ostream& ostream, unw_word_t reg) noexcept {
            constexpr std::size_t address_width = std::numeric_limits<std::uintptr_t>::digits / 4;
            ostream << "0x" << std::setfill('0') << std::setw(address_width) << reg;
        }
        
        char symbol[1024];
        
    } /// namespace (anon.)
    
    void backtrace(std::ostream& ostream) noexcept {
        unw_cursor_t cursor;
        unw_context_t context;
        unw_getcontext(&context);
        unw_init_local(&cursor, &context);
        ostream << std::hex << std::uppercase;
        while (0 < unw_step(&cursor)) {
            unw_word_t ip = 0;
            unw_get_reg(&cursor, UNW_REG_IP, &ip);
            if (ip == 0) { break; }
            unw_word_t sp = 0;
            unw_get_reg(&cursor, UNW_REG_SP, &sp);
            print_reg(ostream, ip);
            ostream << ": (SP:";
            print_reg(ostream, sp);
            ostream << ") ";
            unw_word_t offset = 0;
            if (unw_get_proc_name(&cursor,
                                   symbol, sizeof(symbol),
                                  &offset) == 0) {
                ostream <<     "(" << postprocess(demangle(symbol))
                        << " + 0x" << offset << ")"
                        << std::endl;
            } else {
                ostream << "(unable to get a symbol name for this frame)"
                        << std::endl;
            }
        }
        ostream << std::flush;
    }
    
} /// namespace terminator