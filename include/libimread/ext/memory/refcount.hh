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

namespace memory {
    
    static GuidGenerator generator = GuidGenerator();
    static std::unordered_map<Guid, std::atomic<int64_t>> refcounts;
    
    template <typename Target>
    struct DefaultDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            delete target;
        }
    };
    
    template <typename Target>
    struct ArrayDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            delete[] target;
        }
    };
    
    template <typename Target>
    struct DeallocationDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            free(target);
        }
    };
    
    template <typename Target,
              typename Deleter = DefaultDeleter<Target>>
    struct RefCount {
        
        Guid guid;
        Target *object;
        Deleter deleter;
        
        template <typename ...Args>
        static RefCount MakeRef(Args&& ...args) {
            return RefCount<Target>(
                new Target(std::forward<Args>(args)...));
        }
        
        RefCount() = default;
        
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
            refcounts[guid].store(1);
        }
        
        ~RefCount() {
            release();
        }
        
        void retain() { refcounts[guid]++; }
        void release() { refcounts[guid]--; gc(); }
        
        /// for debugging purposes really
        int64_t retainCount() const {
            return refcounts[guid].load();
        }
        
        RefCount &operator=(const RefCount& other) {
            RefCount(other).swap(*this);
            return *this;
        }
        
        Target* operator->() const { return  object; }
        Target  operator* () const { return *object; }
        
        void gc() {
            if (refcounts[guid].load() < 1) {
                deleter(object);
            }
        }
        
        std::size_t hash() const {
            std::hash<Target*> pHasher;
            std::hash<Guid> guidHasher;
            std::size_t H = guidHasher(guid);
            ::detail::rehash(H, pHasher(object));
            return H;
        }
        
        void swap(RefCount& other) {
            using std::swap;
            guid.swap(other.guid);
            swap(other.object, object);
            swap(other.deleter, deleter);
        }
        
        friend void swap(RefCount& lhs, RefCount& rhs) {
            /// so I guess like 'friend' implies 'static' ?!
            lhs.swap(rhs);
        }
        
    };
    
    /// GARBAGEDAAAAAY
    static void garbageday();
    
}; /* namespace memory */

#endif /// LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_