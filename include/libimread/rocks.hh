/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Adapted from my own Objective-C++ class:
///     https://gist.github.com/fish2000/b3a7d8accae8d046703f728b4ac82009

#ifndef LIBIMREAD_INCLUDE_ROCKS_HH_
#define LIBIMREAD_INCLUDE_ROCKS_HH_

#include <libimread/store.hh>
#include <libimread/ext/memory/refcount.hh>

namespace memory {
    
    void killrocks(void*&);
    
    template <typename Target>
    struct RocksDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const { if (target) { memory::killrocks(target); } }
    };
    
    using rocks_ptr = RefCount<void, RocksDeleter<void>>;
}

namespace store {
    
    class rocks : public stringmapper {
        
        public:
            virtual bool can_store() const noexcept override;
            
            template <typename T>
            T* get_instance() const { return instance.get<T>(); }
        
        public:
            rocks(rocks const&);
            rocks(rocks&&) noexcept;
            explicit rocks(std::string const& filepth);
            virtual ~rocks();
        
        public:
            /// implementation of the stringmapper API, in terms of the RocksDB API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
            
        protected:
            memory::rocks_ptr instance;
            std::string rockspth{ NULL_STR };
        
    };
    
} /// namespace store

#endif /// LIBIMREAD_INCLUDE_ROCKS_HH_
