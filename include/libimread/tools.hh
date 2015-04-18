// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
#define LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012

#include <vector>
#include <string>
#include <memory>
#include <list>
#include <tuple>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace im {
    
    inline const char *split_filename(const char *const filename,
                                      char *const body = 0) {
        if (!filename) {
            if (body) { *body = 0; }
            return 0;
        }
        
        const char *p = 0;
        for (const char *np = filename; np >= filename && (p = np);
             np = std::strchr(np, '.') + 1) {
        }
        if (p == filename) {
            if (body) { std::strcpy(body, filename); }
            return filename + std::strlen(filename);
        }
        
        const unsigned int l = static_cast<unsigned int>(p - filename - 1);
        if (body) {
            std::memcpy(body, filename, l);
            body[l] = 0;
        }
        
        return p;
    }
    
    class MultiEnableShared : public std::enable_shared_from_this<MultiEnableShared> {
        public:
            virtual ~MultiEnableShared() {}
    };
    
    template <typename T>
    class EnableShared : virtual public MultiEnableShared {
        public:
            std::shared_ptr<T> shared_from_this() {
                return std::dynamic_pointer_cast<T>(MultiEnableShared::shared_from_this());
            }
            
            template <typename Down>
            std::shared_ptr<Down> downcast_from_this() {
                return std::dynamic_pointer_cast<Down>(MultiEnableShared::shared_from_this());
            }
    };
    
    template <typename T, typename pT>
    std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT> &&src) {
        /// Force a dynamic_cast upon a unique_ptr via interim swap
        /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
        /// ... from http://stackoverflow.com/a/14777419/298171
        if (!src) { return std::unique_ptr<T>(); }
        
        /// Throws a std::bad_cast() if this doesn't work out
        T *dst = &dynamic_cast<T&>(*src.get());
        
        src.release();
        std::unique_ptr<T> ret(dst);
        return ret;
    }
    
    template <typename T>
    void ptr_swap(T*& oA, T*& oB) {
        T* oT = oA; oA = oB; oB = oT;
    }
    
    template <typename T>
    inline std::vector<T*> allrows(Image &im) {
        std::vector<T*> res;
        const int h = im.dim(0);
        for (int r = 0; r != h; ++r) {
            res.push_back(im.rowp_as<T>(r));
        }
        return res;
    }
    
    inline std::vector<byte> full_data(byte_source &s) {
        std::vector<byte> res;
        byte buffer[4096];
        while (std::size_t n = s.read(buffer, sizeof(buffer))) {
            res.insert(res.end(), buffer, buffer + n);
        }
        return res;
    }
    
    inline uint8_t read8(byte_source &s) {
        byte out;
        if (s.read(&out, 1) != 1) {
            throw CannotReadError("im::read8(): File ended prematurely");
        }
        return out;
    }
    
    inline uint16_t read16_le(byte_source &s) {
        uint8_t b0 = read8(s);
        uint8_t b1 = read8(s);
        return (uint16_t(b1) << 8) | uint16_t(b0);
    }
    
    inline uint32_t read32_le(byte_source &s) {
        uint16_t s0 = read16_le(s);
        uint16_t s1 = read16_le(s);
        return (uint32_t(s1) << 16) | uint32_t(s0);
    }
    
    struct stack_based_memory_pool {
        // This class manages an allocator which releases all allocated memory on
        // stack exit
        stack_based_memory_pool() { }
        
        ~stack_based_memory_pool() {
            for (unsigned i = 0; i != data.size(); ++i) {
                operator delete(data[i]);
                data[i] = 0;
            }
        }
        
        void *allocate(const int n) {
            data.reserve(data.size() + 1);
            void *d = operator new(n);
            data.push_back(d);
            return d;
        }
        
        template <typename T>
        T allocate_as(const int n) {
            return static_cast<T>(this->allocate(n));
        }
        
        private:
            std::vector<void*> data;
    };

}

#endif // LPC_TOOLS_H_INCLUDE_GUARD_WED_FEB__8_17_05_13_WET_2012
