/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_
#define LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_

#include <unordered_map>
#include <type_traits>
#include <functional>
#include <utility>
#include <atomic>
#include <thread>
#include <cstddef>
#include <cstdint>

#include <guid.h>
#include <libimread/libimread.hpp>
#include <libimread/rehash.hh>

namespace memory {
    
    using refcount_t = std::unordered_map<Guid, std::atomic<int64_t>>;
    using thread_t = std::thread;
    
    extern GuidGenerator generator;
    extern refcount_t refcounts;
    // extern thread_t garbagecollector;
    
    template <typename Target>
    struct DefaultDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            if (target) {
                delete target;
                target = nullptr;
            }
        }
    };
    
    template <typename Target>
    struct ArrayDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            if (target) {
                delete[] target;
                target = nullptr;
            }
        }
    };
    
    template <typename Target>
    struct DeallocationDeleter : public std::unary_function<std::add_pointer_t<Target>, void> {
        using target_ptr_t = typename std::add_pointer_t<Target>;
        void operator()(target_ptr_t target) const {
            if (target) {
                std::free(target);
                target = nullptr;
            }
        }
    };
    
    template <typename Target,
              typename Deleter = std::conditional_t<
                                 std::is_same<Target, void>::value,
                                     DeallocationDeleter<Target>,
                                     DefaultDeleter<Target>>>
    struct RefCount {
        
        mutable Guid guid;
        mutable Target* object;
        Deleter deleter;
        
        template <typename ...Args>
        static RefCount MakeRef(Args&& ...args) {
            return RefCount<Target>(
                new Target(std::forward<Args>(args)...));
        }
        
        template <typename ...Args>
        static RefCount Allocate(Args&& ...args) {
            return RefCount<void>(
                std::malloc(std::forward<Args>(args)...));
        }
        
        RefCount()
            :object(nullptr)
            ,deleter(Deleter{})
            {
                init();
            }
        
        explicit RefCount(Target* o)
            :object(o), deleter(Deleter{})
            {
                init();
                retain();
            }
        
        explicit RefCount(Target* o, Deleter d)
            :object(o), deleter(d)
            {
                init();
                retain();
            }
        
        RefCount(RefCount const& other)
            :object(other.object)
            ,guid(other.guid)
            ,deleter(other.deleter)
            {
                retain();
            }
        
        RefCount(RefCount&& other) noexcept
            :object(std::move(other.object))
            ,guid(std::move(other.guid))
            ,deleter(std::move(other.deleter))
            {
                retain();
                other.release();
            }
        
        void init() {
            guid = generator.newGuid();
            refcounts[guid].store(0);
        }
        
        virtual ~RefCount() {
            release();
        }
        
        void retain() {     refcounts[guid]++;          }
        void release() {    refcounts[guid]--;  gc();   }
        
        /// for debugging purposes really
        int64_t retainCount() const {
            return refcounts[guid].load();
        }
        
        RefCount& operator=(RefCount const& other) {
            if (guid != other.guid) {
                RefCount(other).swap(*this);
            }
            return *this;
        }
        
        Target* operator->() const { return  object; }
        Target  operator* () const { return *object; }
        
        Target* get() const { return object; }
        
        template <typename T>
        T* get() const { return static_cast<T*>(object); }
        
        void reset(Target* reset_to = nullptr) {
            if (object) {
                deleter(object);
                refcounts[guid].store(0);
            }
            object = reset_to;
            if (object) {
                retain();
            }
        }
        
        void gc() {
            if (refcounts[guid].load() < 1) {
                deleter(object);
            }
        }
        
        std::size_t hash() const {
            std::hash<Target*> pHasher;
            std::hash<Guid> guidHasher;
            std::size_t H = guidHasher(guid);
            hash::rehash(H, pHasher(object));
            return H;
        }
        
        void swap(RefCount& other) {
            using std::swap;
            guid.swap(other.guid);
            swap(other.object,  object);
            swap(other.deleter, deleter);
        }
        
        friend void swap(RefCount& lhs, RefCount& rhs) {
            /// so I guess like 'friend' implies 'static' ?!
            lhs.swap(rhs);
        }
        
    };
    
    /// GARBAGEDAAAAAY
    extern void garbageday();
    
}; /* namespace memory */

#endif /// LIBIMREAD_EXT_MEMORY_REFCOUNT_HH_