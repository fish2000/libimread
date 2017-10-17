/// Copyright 2014-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/ext/pickle/encoder.hh>
#include <libimread/ext/pickle/opcodes.hh>

namespace store {
    
    namespace pickle {
        
        /// OPCODE METHODS: op_MARK() sends the opcode for MARK, etc.
        /// Method delcarations/definitions generated with regex: ([A-Z0-9_]+)\s+= '(\S+)',$
        
        /// define the opcode emitter methods:
        #define OPCODE(__name__, __char__) void encoder::op_##__name__() { emit(__char__); }
        OPCODE(MARK,            '(');
        OPCODE(STOP,            '.');
        OPCODE(POP,             '0');
        OPCODE(POP_MARK,        '1');
        OPCODE(DUP,             '2');
        OPCODE(FLOAT,           'F');
        OPCODE(INT,             'I');
        OPCODE(BININT,          'J');
        OPCODE(BININT1,         'K');
        OPCODE(LONG,            'L');
        OPCODE(BININT2,         'M');
        OPCODE(NONE,            'N');
        OPCODE(PERSID,          'P');
        OPCODE(BINPERSID,       'Q');
        OPCODE(REDUCE,          'R');
        OPCODE(STRING,          'S');
        OPCODE(BINSTRING,       'T');
        OPCODE(SHORT_BINSTRING, 'U');
        OPCODE(UNICODE,         'V');
        OPCODE(BINUNICODE,      'X');
        OPCODE(APPEND,          'a');
        OPCODE(BUILD,           'b');
        OPCODE(GLOBAL,          'c');
        OPCODE(DICT,            'd');
        OPCODE(EMPTY_DICT,      '}');
        OPCODE(APPENDS,         'e');
        OPCODE(GET,             'g');
        OPCODE(BINGET,          'h');
        OPCODE(INST,            'i');
        OPCODE(LONG_BINGET,     'j');
        OPCODE(LIST,            'l');
        OPCODE(EMPTY_LIST,      ']');
        OPCODE(OBJ,             'o');
        OPCODE(PUT,             'p');
        OPCODE(BINPUT,          'q');
        OPCODE(LONG_BINPUT,     'r');
        OPCODE(SETITEM,         's');
        OPCODE(TUPLE,           't');
        OPCODE(EMPTY_TUPLE,     ')');
        OPCODE(SETITEMS,        'u');
        OPCODE(BINFLOAT,        'G');
        OPCODE(PROTO,           '\x80');
        OPCODE(NEWOBJ,          '\x81');
        OPCODE(EXT1,            '\x82');
        OPCODE(EXT2,            '\x83');
        OPCODE(EXT4,            '\x84');
        OPCODE(TUPLE1,          '\x85');
        OPCODE(TUPLE2,          '\x86');
        OPCODE(TUPLE3,          '\x87');
        OPCODE(NEWTRUE,         '\x88');
        OPCODE(NEWFALSE,        '\x89');
        OPCODE(LONG1,           '\x8a');
        OPCODE(LONG4,           '\x8b');
        OPCODE(BINBYTES,        'B');
        OPCODE(SHORT_BINBYTES,  'C');
        OPCODE(SHORT_BINUNICODE, '\x8c');
        OPCODE(BINUNICODE8,     '\x8d');
        OPCODE(BINBYTES8,       '\x8e');
        OPCODE(EMPTY_SET,       '\x8f');
        OPCODE(ADDITEMS,        '\x90');
        OPCODE(FROZENSET,       '\x91');
        OPCODE(NEWOBJ_EX,       '\x92');
        OPCODE(STACK_GLOBAL,    '\x93');
        OPCODE(MEMOIZE,         '\x94');
        OPCODE(FRAME,           '\x95');
        #undef OPCODE
        
        /// convenience emitter methods
        void encoder::newline() { emit('\n'); }
        void encoder::zero()    { emit('0'); }
        void encoder::one()     { emit('1'); }
        
        void encoder::encode(std::nullptr_t) {
            /// NO-OP
        }
        
        void encoder::encode(void) {
            /// NO-OP
        }
        
        void encoder::encode(void* operand) {
            /// ???
        }
        
        void encoder::encode(bool operand) {
            op_INT();
            zero();
            if (operand) {
                one();
            } else {
                zero();
            }
            newline();
        }
        
        void encoder::encode(std::size_t operand) {
            op_LONG();
            emit(std::to_string(operand) + "L");
            newline();
        }
        
        void encoder::encode(ssize_t operand) {
            op_LONG();
            emit(std::to_string(operand) + "L");
            newline();
        }
        
        void encoder::encode(int8_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(int16_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(int32_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(int64_t operand) {
            op_LONG();
            emit(std::to_string(operand) + "L");
            newline();
        }
        
        void encoder::encode(uint8_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(uint16_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(uint32_t operand) {
            op_INT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(uint64_t operand) {
            op_LONG();
            emit(std::to_string(operand) + "L");
            newline();
        }
        
        void encoder::encode(float operand) {
            op_FLOAT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(double operand) {
            op_FLOAT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(long double operand) {
            op_FLOAT();
            emit(std::to_string(operand));
            newline();
        }
        
        void encoder::encode(char* operand) {
            op_STRING();
            emit("'" + std::string(operand) + "'");
            newline();
        }
        
        void encoder::encode(char const* operand) {
            op_STRING();
            emit("'" + std::string(operand) + "'");
            newline();
        }
        
        void encoder::encode(std::string const& operand) {
            op_STRING();
            emit("'" + operand + "'");
            newline();
        }
        
        void encoder::encode(char* operand, std::size_t siz) {
            op_STRING();
            emit("'" + std::string(operand, siz) + "'");
            newline();
        }
        
        void encoder::encode(char const* operand, std::size_t siz) {
            op_STRING();
            emit("'" + std::string(operand, siz) + "'");
            newline();
        }
        
        void encoder::encode(std::string const& operand, std::size_t siz) {
            op_STRING();
            emit("'" + operand.substr(siz) + "'");
            newline();
        }
        
        void encoder::encode(std::wstring const& operand) {
            op_BINUNICODE();
            emit("'" + operand + "'");
            newline();
        }
        
        void encoder::encode(std::wstring const& operand, std::size_t siz) {
            op_BINUNICODE();
            emit("'" + operand.substr(siz) + "'");
            newline();
        }
        
        
        
    } /// namespace pickle
    
} /// namespace store