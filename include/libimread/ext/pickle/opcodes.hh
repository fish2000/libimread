/// Copyright 2014-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_PICKLE_OPCODES_HH_
#define LIBIMREAD_EXT_PICKLE_OPCODES_HH_

/* Pickle opcodes. These must be kept updated with pickle.py.
   Extensive docs are in pickletools.py.
   Lifted directly from the 3.x branch of the cPython source:
   https://github.com/python/cpython/blob/191e3138200906e43cba9347177914325b54843f/Modules/_pickle.c#L29-L103
*/

namespace store {
    
    namespace pickle {
        
        enum {
            HIGHEST_PROTOCOL = 4,
            DEFAULT_PROTOCOL = 3
        };
        
        enum class opcode {
            MARK            = '(',
            STOP            = '.',
            POP             = '0',
            POP_MARK        = '1',
            DUP             = '2',
            FLOAT           = 'F',
            INT             = 'I',
            BININT          = 'J',
            BININT1         = 'K',
            LONG            = 'L',
            BININT2         = 'M',
            NONE            = 'N',
            PERSID          = 'P',
            BINPERSID       = 'Q',
            REDUCE          = 'R',
            STRING          = 'S',
            BINSTRING       = 'T',
            SHORT_BINSTRING = 'U',
            UNICODE         = 'V',
            BINUNICODE      = 'X',
            APPEND          = 'a',
            BUILD           = 'b',
            GLOBAL          = 'c',
            DICT            = 'd',
            EMPTY_DICT      = '}',
            APPENDS         = 'e',
            GET             = 'g',
            BINGET          = 'h',
            INST            = 'i',
            LONG_BINGET     = 'j',
            LIST            = 'l',
            EMPTY_LIST      = ']',
            OBJ             = 'o',
            PUT             = 'p',
            BINPUT          = 'q',
            LONG_BINPUT     = 'r',
            SETITEM         = 's',
            TUPLE           = 't',
            EMPTY_TUPLE     = ')',
            SETITEMS        = 'u',
            BINFLOAT        = 'G',
            
            /* Protocol 2. */
            PROTO       = '\x80',
            NEWOBJ      = '\x81',
            EXT1        = '\x82',
            EXT2        = '\x83',
            EXT4        = '\x84',
            TUPLE1      = '\x85',
            TUPLE2      = '\x86',
            TUPLE3      = '\x87',
            NEWTRUE     = '\x88',
            NEWFALSE    = '\x89',
            LONG1       = '\x8a',
            LONG4       = '\x8b',
            
            /* Protocol 3 (Python 3.x) */
            BINBYTES       = 'B',
            SHORT_BINBYTES = 'C',
            
            /* Protocol 4 */
            SHORT_BINUNICODE = '\x8c',
            BINUNICODE8      = '\x8d',
            BINBYTES8        = '\x8e',
            EMPTY_SET        = '\x8f',
            ADDITEMS         = '\x90',
            FROZENSET        = '\x91',
            NEWOBJ_EX        = '\x92',
            STACK_GLOBAL     = '\x93',
            MEMOIZE          = '\x94',
            FRAME            = '\x95'
        
        }; /// enum class opcode
        
        enum {
            BATCHSIZE = 1000,
            FAST_NESTING_LIMIT = 50,
            WRITE_BUF_SIZE = 4096,
            PREFETCH = 8192 * 16,
            FRAME_SIZE_TARGET = 64 * 1024,
            FRAME_HEADER_SIZE = 9
        };
        
    } /// namespace pickle
    
} /// namespace store

#endif /// LIBIMREAD_EXT_PICKLE_OPCODES_HH_