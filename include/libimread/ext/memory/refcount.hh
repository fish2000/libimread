/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_
#define LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_

#include <unordered_map>
#include <type_traits>
#include <functional>
#include <utility>
#include <atomic>

#include <cstddef>
#include <cstdint>

#include <guid.h>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>

namespace std {
    
    template <>
    void swap(Guid& guid0, Guid& guid1);
    
    template <>
    struct hash<Guid> {
        
        typedef Guid argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& guid) const {
            std::hash<std::string> hasher;
            return static_cast<result_type>(hasher(guid.str()));
        }
        
    };
    
}; /* namespace std */

namespace memory {
    
    static GuidGenerator generator = GuidGenerator();
    static std::unordered_map<Guid, std::atomic<int64_t>> refcounts;
    
    template <typename Target>
    struct DefaultDeleter : public std::unary_function<Target*, void> {
        using Base = std::unary_function<Target*, void>;
        void operator()(Target* target) const {
            /// default deleter
            WTF("DefaultDeleter::operator()(Target*): about to call `delete target`");
            delete target;
        }
    };
    
    template <typename Target,
              typename Deleter = DefaultDeleter<Target>>
    struct RefCount : public DefaultDeleter<Target>::Base {
        
        //using Deleter = std::add_pointer_t<void(const Target*)>;
        // using Deleter = void(Target*);
        
        Guid guid;
        Target *object;
        Deleter deleter;
        
        template <typename ...Args>
        static RefCount MakeRef(Args&& ...args) {
            return RefCount<Target>(new Target(std::forward<Args>(args)...));
        }
        
        explicit RefCount(Target *o)
            :object(o), deleter(Deleter{})
            {
                init();
                retain();
            }
        explicit RefCount(Target *o, Deleter d)
            :object(o), deleter(d)
            {
                init();
                retain();
            }
        
        RefCount(const RefCount& other)
            :object(other.object)
            ,guid(other.guid)
            {
                retain();
            }
        
        void init() {
            guid = generator.newGuid();
            refcounts[guid].store(0);
            // if (deleter == nullptr) {
            //     deleter = (Deleter)this; /// default
            // }
        }
        
        virtual ~RefCount() {
            WTF("RefCount::~RefCount(): in destructor");
            release();
        }
        
        void retain() { refcounts[guid]++; }
        void release() { refcounts[guid]--; gc(); }
        
        /// for debugging purposes really
        int64_t retainCount() { return refcounts[guid].load(); }
        
        RefCount &operator=(const RefCount& other) {
            RefCount(other).swap(*this);
            return *this;
        }
        RefCount &operator=(RefCount&& other) {
            other.swap(*this);
            return *this;
        }
        
        void operator()(Target* target) const {
            /// default deleter
            delete target;
        }
        
        inline void gc() {
            WTF("Collecting garbage, refcount[guid] = ", refcounts[guid].load());
            if (refcounts[guid].load() < 1) {
                deleter(object);
            }
        }
        
        void swap(RefCount& other) noexcept {
            std::swap(other.guid, guid);
            std::swap(other.object, object);
            std::swap(other.deleter, deleter);
        }
        
    };
    
    /// GARBAGEDAAAAAY!
    static void garbageday();
    
}; /* namespace memory */

#endif /// LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_