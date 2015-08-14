/*
 * fmem.c : fmemopen() on top of BSD's funopen()
 * 20081017 AF (orig)
 * 20150115 Alexander Fucking Bohn (this totally awesome c++11'd iteration)
 */

#include <libimread/ext/memory/fmemopen.hh>

namespace memory {
    
    struct fmem {
        std::size_t pos;
        std::size_t size;
        char *buffer;
    };
    
    typedef struct fmem fmem_t;
    
    int readfn(void *handler, char *buf, int size) {
        int count = 0;
        fmem_t *mem = (fmem_t *)handler;
        std::size_t available = mem->size - mem->pos;
        if (size > available) { size = available; }
        for (count = 0; count < size; mem->pos++, count++) {
            buf[count] = mem->buffer[mem->pos];
        }
        return count;
    }
    
    int writefn(void *handler, const char *buf, int size) {
        int count = 0;
        fmem_t *mem = (fmem_t *)handler;
        std::size_t available = mem->size - mem->pos;
        if (size > available) { size = available; }
        for (count=0; count < size; mem->pos++, count++) {
            mem->buffer[mem->pos] = buf[count];
        }
        return count; // ? count : size;
    }
    
    fpos_t seekfn(void *handler, fpos_t offset, int whence) {
        std::size_t pos;
        fmem_t *mem = (fmem_t *)handler;
        switch (whence) {
            case SEEK_SET: pos = offset; break;
            case SEEK_CUR: pos = mem->pos + offset; break;
            case SEEK_END: pos = mem->size + offset; break;
            default: return -1;
        }
        if (pos > mem->size) { return -1; }
        mem->pos = pos;
        return (fpos_t)pos;
    }
    
    int closefn(void *handler) {
        free(handler);
        return 0;
    }
    
    /* simple, but portable version of fmemopen for OS X / BSD */
    FILE *fmemopen(void *buf, std::size_t size, const char *mode) {
        fmem_t *mem = (fmem_t *)::malloc(sizeof(fmem_t));
        
        std::memset(mem, 0, sizeof(fmem_t));
        mem->size = size, mem->buffer = (char *)buf;
        return ::funopen(mem, readfn, writefn, seekfn, closefn);
    }
    
    buffer source(void *buf, std::size_t size) {
        return buffer(fmemopen(buf, size, "rb"));
    }
    
    buffer sink(void *buf, std::size_t size) {
        return buffer(fmemopen(buf, size, "wb"));
    }
    

} /// namespace memory
    