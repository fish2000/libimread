/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_
#define LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_

#include <mutex>
#include <string>
#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/resolver.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>

/// Welcome to Singleton, USA

namespace im {
        
    class Base : public std::enable_shared_from_this<Base> {
        
        using shared_t = std::shared_ptr<Base>;
        
        private:
            static std::mutex wintermute;
            Base(const Base& copy) = delete;
            Base &operator=(const Base& copy) = delete;
        
        protected:
            constexpr Base() noexcept = default;
            virtual ~Base() {}
            virtual shared_t shared() {
                return shared_from_this();
            }
        
        public:
            
            static shared_t& get() {
                static shared_t instance = nullptr;
                if (!instance) {
                    std::lock_guard<std::mutex> lock(wintermute);
                    if (!instance) { instance.reset(new Base()); }
                    return instance;
                }
                return shared();
            }
            
            virtual bool has(const path&) = 0;
            virtual bool has(path&&) = 0;
            
    };
    
    class Singleton : public Base {
        public:
            virtual bool has(const std::string& s) { return true; }
            virtual bool has(std::string&& s) { return true; }
        protected:
            constexpr Singleton() noexcept = default;
            virtual ~Singleton() {}
    };
}

#endif /// LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_