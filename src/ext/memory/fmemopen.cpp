/*
 * fmem.c : fmemopen() on top of BSD's funopen()
 * 20081017 AF (orig)
 * 20150115 Alexander Fucking Bohn (this totally awesome c++11'd iteration)
 */

#include <cstdlib>
#include <cstring>
#include <functional>
#include <libimread/ext/memory/fmemopen.hh>

namespace memory {
    
    namespace impl {
        
        struct fmem {
            std::size_t pos;
            std::size_t size;
            char* buffer;
        };
        
        typedef struct fmem fmem_t;
        
        int readfn(void* handler, char* buf, int size) {
            fmem_t* mem = reinterpret_cast<fmem_t*>(handler);
            std::size_t available = mem->size - mem->pos;
            if (size > available) { size = available; }
            std::memcpy(buf, mem->buffer + mem->pos, size);
            mem->pos += size;
            return size;
        }
        
        int writefn(void* handler, const char* buf, int size) {
            fmem_t* mem = reinterpret_cast<fmem_t*>(handler);
            std::size_t available = mem->size - mem->pos;
            if (size > available) { size = available; }
            std::memcpy(mem->buffer + mem->pos, buf, size);
            mem->pos += size;
            return size;
        }
        
        fpos_t seekfn(void* handler, fpos_t offset, int whence) {
            std::size_t pos;
            fmem_t* mem = reinterpret_cast<fmem_t*>(handler);
            switch (whence) {
                case SEEK_SET:
                    pos = offset;
                    break;
                case SEEK_CUR:
                    pos = mem->pos + offset;
                    break;
                case SEEK_END:
                    pos = mem->size + offset;
                    break;
                default:
                    return -1;
            }
            if (pos > mem->size) { return -1; }
            mem->pos = pos;
            return static_cast<fpos_t>(pos);
        }
        
        int closefn(void* handler) {
            delete reinterpret_cast<fmem_t*>(handler);
            return 0;
        }
        
    };
    
    /// simple but portable version of fmemopen() for OS X / BSD
    FILE* fmemopen(void* buf, std::size_t size, const char* mode) {
        return ::funopen(
            new impl::fmem_t{ 0, size, static_cast<char*>(buf) },
            impl::readfn, impl::writefn,
            impl::seekfn, impl::closefn);
    }
    
    buffer source(void* buf, std::size_t size) {
        return buffer(fmemopen(buf, size, "rb"));
    }
    
    buffer sink(void* buf, std::size_t size) {
        return buffer(fmemopen(buf, size, "wb"));
    }
    

} /// namespace memory
    