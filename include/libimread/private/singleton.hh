/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_
#define LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_

#include <mutex>
#include <string>
#include <memory>
#include <libimread/libimread.hpp>
// #include <libimread/ext/filesystem/path.h>
// #include <libimread/ext/filesystem/resolver.h>
// #include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>

/// Welcome to Singleton, USA

namespace filesystem {
    class path;
}

namespace single {
        
    class base : public std::enable_shared_from_this<base> {
        
        using shared_t = std::shared_ptr<Base>;
        
        private:
            static std::mutex wintermute;
            base(base const&) = delete;
            base& operator=(base const&) = delete;
        
        protected:
            constexpr base() noexcept = default;
            virtual ~base() {}
            virtual shared_t shared() {
                return shared_from_this();
            }
        
        public:
            static shared_t& get() {
                static shared_t instance = nullptr;
                if (!instance) {
                    std::lock_guard<std::mutex> lock(wintermute);
                    if (!instance) { instance.reset(new base()); }
                    return instance;
                }
                return shared();
            }
        
        public:
            virtual bool has(std::string const&) = 0;
            virtual bool has(std::string&&) = 0;
            
    };
    
    class singleton : public base {
        public:
            virtual bool has(std::string const& s) { return true; }
            virtual bool has(std::string&& s) { return true; }
        protected:
            constexpr singleton() noexcept = default;
            virtual ~singleton() {}
    };
}

#endif /// LIBIMREAD_INCLUDE_PRIVATE_SINGLETON_HH_