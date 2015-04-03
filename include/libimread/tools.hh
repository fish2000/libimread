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
#include <libimread/base.hh>

namespace im {
    
    namespace pre {
        
        /// based on variadic_capture by Scott Schurr
        template <typename... Types>
        class Store {
            private:
                using Storage = std::tuple<Types...>;
                Storage storage;
            public:
                void put(Types... values) {
                    storage = std::make_tuple(values...);
                }
                constexpr std::size_t count() const {
                    return sizeof...(Types);
                }
                template <std::size_t idx>
                auto get() -> const typename std::tuple_element<idx, Storage>::type& {
                    return (std::get<idx>(storage));
                }
                template <std::size_t idx>
                constexpr auto preget() -> const typename std::tuple_element<idx, Storage>::type& {
                    return (std::get<idx>(storage));
                }
        };
        
        /// based on str_const by Scott Schurr
        constexpr unsigned int inrange(unsigned int idx, unsigned int len) {
            return idx >= len ? throw std::out_of_range("n/a") : idx;
        }
        
        template <unsigned N>
        constexpr char nth(const char (&chars)[N], unsigned int i) {
            return inrange(i, N-1), chars[i];
        }
        
        constexpr bool static_compare(const char *a, const char *b) {
            return *a == *b && (*a == '\0' || static_compare(a + 1, b + 1));
        }
        
        class String {
        private:
                const char *cstr;
                const std::size_t csiz;
                static constexpr std::size_t fnv_prime = (
                    sizeof(std::size_t) == 8 ? 1099511628211u : 16777619u);
                static constexpr std::size_t fnv_offset = (
                    sizeof(std::size_t) == 8 ? 14695981039346656037u : 2166136261u);
            public:
                template <std::size_t N>
                constexpr String(const char (&a)[N])
                    : cstr(a), csiz(N-1)
                    {}
                constexpr String(const char* a, std::size_t Nn)
                    : cstr(a), csiz(Nn-1)
                    {}
                constexpr String()
                    : cstr(""), csiz(0)
                    {}
                constexpr operator const char*() { return cstr; }
                constexpr char operator[](std::size_t n) {
                    return n < csiz ? cstr[n] : throw std::out_of_range("n/a");
                }
                constexpr bool operator==(String const& rhs) {
                    return static_compare(cstr, rhs.c_str());
                }
                constexpr bool operator==(const char *rhs) {
                    return static_compare(cstr, rhs);
                }
                constexpr std::size_t size() const { return csiz; }
                constexpr char first() const { return cstr[0]; }
                constexpr char last() const { return cstr[csiz-1]; }
                constexpr std::size_t end() const { return csiz; }
                constexpr std::size_t begin() const { return 0; }
                constexpr const char *c_str() const { return cstr; }
                constexpr std::size_t fnv_1a_hash(unsigned int i) const {
                    return (i == 0 ? fnv_offset : ((fnv_1a_hash(i-1) ^ cstr[i]) * fnv_prime));
                }
        };
    }
    
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
        while (int n = s.read(buffer, sizeof(buffer))) {
            res.insert(res.end(), buffer, buffer + n);
        }
        return res;
    }
    
    inline uint8_t read8(byte_source &s) {
        byte out;
        if (s.read(&out, 1) != 1) {
            throw CannotReadError("File ended prematurely");
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
